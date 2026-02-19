#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request
from datetime import datetime

import ee
import pandas as pd


DEFAULT_TRAINING_CSV = "data/water_quality_training_dataset.csv"
DEFAULT_SUBMISSION_CSV = "data/submission_template.csv"
DEFAULT_OUTPUT_DIR = "data/modis_patches"
DEFAULT_EE_PROJECT = "gearth-486818"
DEFAULT_SCALE = 500
DEFAULT_PATCH_SIZE_PX = 100
DEFAULT_MAX_DAYS_SEARCH = 32
DEFAULT_BATCH_SIZE = 100
DEFAULT_DOWNLOAD_WORKERS = 8
DEFAULT_DOWNLOAD_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.5
DATE_FMT = "%d-%m-%Y"

MODIS_REFLECTANCE = "MODIS/061/MCD43A4"
MODIS_QUALITY = "MODIS/061/MCD43A2"

# Output band order chosen to be compatible with local index computation:
# Green, Red, NIR, SWIR1, SWIR2.
REFL_BANDS = [
    "Nadir_Reflectance_Band4",  # Green
    "Nadir_Reflectance_Band1",  # Red
    "Nadir_Reflectance_Band2",  # NIR
    "Nadir_Reflectance_Band6",  # SWIR1
    "Nadir_Reflectance_Band7",  # SWIR2
]
OUT_BANDS = ["green", "red", "nir", "swir1", "swir2"]
QA_BANDS = [
    "BRDF_Albedo_Band_Quality_Band4",
    "BRDF_Albedo_Band_Quality_Band1",
    "BRDF_Albedo_Band_Quality_Band2",
    "BRDF_Albedo_Band_Quality_Band6",
    "BRDF_Albedo_Band_Quality_Band7",
]


def initialize_ee(project: str, use_high_volume: bool) -> None:
    if use_high_volume:
        ee_url = "https://earthengine-highvolume.googleapis.com"
        try:
            ee.Initialize(project=project, opt_url=ee_url)
            return
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project, opt_url=ee_url)
            return

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def parse_date_iso(date_value: str) -> str:
    return datetime.strptime(str(date_value), DATE_FMT).strftime("%Y-%m-%d")


def get_patch_geometry(lat: float, lon: float, patch_size_px: int, scale: int) -> ee.Geometry:
    side_meters = max(1, int(patch_size_px)) * max(1, int(scale))
    half_side = side_meters / 2.0
    return ee.Geometry.Point([lon, lat]).buffer(half_side).bounds()


def _joined_collection(start_date: ee.Date, end_date: ee.Date, region: ee.Geometry) -> ee.ImageCollection:
    refl = (
        ee.ImageCollection(MODIS_REFLECTANCE)
        .filterDate(start_date, end_date)
        .filterBounds(region)
    )
    qa = (
        ee.ImageCollection(MODIS_QUALITY)
        .filterDate(start_date, end_date)
        .filterBounds(region)
    )
    joined = ee.Join.inner().apply(
        refl,
        qa,
        ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def to_masked_image(match):
        refl_img = ee.Image(match.get("primary"))
        qa_img = ee.Image(match.get("secondary"))
        quality_ok = (
            qa_img.select(QA_BANDS[0]).lte(1)
            .And(qa_img.select(QA_BANDS[1]).lte(1))
            .And(qa_img.select(QA_BANDS[2]).lte(1))
            .And(qa_img.select(QA_BANDS[3]).lte(1))
            .And(qa_img.select(QA_BANDS[4]).lte(1))
        )
        raw = (
            refl_img.select(REFL_BANDS)
            .rename(OUT_BANDS)
            .multiply(0.0001)
            .updateMask(quality_ok)
            .toFloat()
        )
        return raw.copyProperties(refl_img, refl_img.propertyNames())

    return ee.ImageCollection(joined.map(to_masked_image))


def find_nearest_modis_image(
    lat: float,
    lon: float,
    target_date_iso: str,
    patch_size_px: int,
    scale: int,
    max_days_search: int,
):
    target_date = ee.Date(target_date_iso)
    start_date = target_date.advance(-int(max_days_search), "day")
    end_date = target_date.advance(int(max_days_search) + 1, "day")
    region = get_patch_geometry(lat, lon, patch_size_px, scale)

    coll = _joined_collection(start_date, end_date, region).map(
        lambda img: img.set(
            "days_diff", ee.Number(img.date().difference(target_date, "day")).abs()
        )
    )
    has_images = coll.size().gt(0)
    best = ee.Image(
        ee.Algorithms.If(
            has_images,
            coll.sort("days_diff").first(),
            ee.Image.constant([0, 0, 0, 0, 0]).rename(OUT_BANDS).toFloat().set(
                {"system:time_start": target_date.millis(), "days_diff": -1}
            ),
        )
    )

    metadata = ee.Dictionary(
        {
            "has_images": has_images,
            "actual_date": ee.Algorithms.If(
                has_images, best.date().format("YYYY-MM-dd"), "NO_IMAGE"
            ),
            "days_diff": ee.Algorithms.If(
                has_images, ee.Number(best.get("days_diff")), -1
            ),
            "search_start": start_date.format("YYYY-MM-dd"),
            "search_end_exclusive": end_date.format("YYYY-MM-dd"),
            "candidate_count": coll.size(),
        }
    )
    return best.clip(region), region, metadata


def _load_existing_metadata(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            records = json.load(f)
        return {int(item["dataset_index"]): item for item in records}
    except Exception:
        return {}


def _save_metadata(path_json: str, path_csv: str, records: list) -> None:
    records_sorted = sorted(records, key=lambda x: int(x["dataset_index"]))
    with open(path_json, "w") as f:
        json.dump(records_sorted, f, indent=2)
    pd.DataFrame(records_sorted).to_csv(path_csv, index=False)


def _upsert_existing_file_metadata(
    existing_by_id: dict,
    dataset_name: str,
    dataset_index: int,
    patch_path: str,
    scale: int,
    patch_size_px: int,
    row: pd.Series | None = None,
) -> None:
    prev = existing_by_id.get(dataset_index, {})
    target_date_iso = prev.get("target_date_iso")
    target_date = prev.get("target_date")
    lat = prev.get("lat")
    lon = prev.get("lon")

    if row is not None:
        try:
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
        except Exception:
            pass
        try:
            target_date_iso = parse_date_iso(row["Sample Date"])
            target_date = datetime.strptime(target_date_iso, "%Y-%m-%d").strftime("%d-%m-%Y")
        except Exception:
            pass

    existing_by_id[dataset_index] = {
        "dataset": dataset_name,
        "dataset_index": dataset_index,
        "patch_id": dataset_index,
        "file": patch_path,
        "lat": lat,
        "lon": lon,
        "target_date_iso": target_date_iso,
        "target_date": target_date,
        "actual_date": prev.get("actual_date"),
        "days_difference": prev.get("days_difference"),
        "modis_scale_meters": int(scale),
        "patch_size_px": int(patch_size_px),
        "bands": OUT_BANDS,
        "qa_policy": "band_quality<=1 for bands [4,1,2,6,7]",
        "candidate_count": prev.get("candidate_count"),
        "search_start": prev.get("search_start"),
        "search_end_exclusive": prev.get("search_end_exclusive"),
        "download_status": "exists",
        "download_message": "resumed_existing_file",
    }


def _download_patch_local(
    sample: dict,
    patch_size_px: int,
    scale: int,
    max_retries: int,
    initial_backoff_seconds: float,
):
    out_path = sample["file_path"]
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return sample["dataset_index"], True, "exists"

    params = {
        "name": sample["patch_name"],
        "region": sample["region"],
        "dimensions": f"{patch_size_px}x{patch_size_px}",
        "filePerBand": False,
        "format": "GEO_TIFF",
    }

    backoff = initial_backoff_seconds
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            url = sample["image"].getDownloadURL(params)
            req = urllib.request.Request(url, headers={"User-Agent": "awqp-modis/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                content = resp.read()
            with open(out_path, "wb") as f:
                f.write(content)
            return sample["dataset_index"], True, f"downloaded(attempt={attempt})"
        except Exception as exc:
            last_err = str(exc)
            time.sleep(min(backoff, 30))
            backoff *= 2
    return sample["dataset_index"], False, f"failed: {last_err}"


def process_dataset(
    dataset_name: str,
    csv_path: str,
    output_dir: str,
    scale: int,
    patch_size_px: int,
    max_days_search: int,
    batch_size: int,
    download_workers: int,
    download_retries: int,
    initial_backoff_seconds: float,
    start_index: int,
    num_samples: int | None,
) -> None:
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    metadata_json = os.path.join(dataset_dir, "metadata.json")
    metadata_csv = os.path.join(dataset_dir, "metadata.csv")

    existing_by_id = _load_existing_metadata(metadata_json)

    df = pd.read_csv(csv_path)
    if num_samples is None:
        end_index = len(df)
    else:
        end_index = min(start_index + num_samples, len(df))
    df_subset = df.iloc[start_index:end_index]

    print("\n" + "=" * 90)
    print(f"MODIS PATCH DOWNLOAD [{dataset_name}]")
    print("=" * 90)
    print(f"CSV: {csv_path}")
    print(f"Rows: {start_index}..{end_index - 1} ({len(df_subset)})")
    print(f"Patch size: {patch_size_px}x{patch_size_px} pixels")
    print(f"Scale: {scale}m | Search window: ±{max_days_search} days")

    samples_to_download = []
    total_processed = 0
    total_found = 0
    resumed_existing = 0
    num_batches = (len(df_subset) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        b_start = batch_num * batch_size
        b_end = min(b_start + batch_size, len(df_subset))
        batch_df = df_subset.iloc[b_start:b_end]
        print(f"\n  Batch {batch_num + 1}/{num_batches}: rows {b_start}..{b_end - 1}")

        prepared = []
        meta_query = {}
        for idx, row in batch_df.iterrows():
            patch_name = f"patch_{idx}"
            patch_path = os.path.join(dataset_dir, f"{patch_name}.tif")
            if os.path.exists(patch_path) and os.path.getsize(patch_path) > 0:
                resumed_existing += 1
                total_processed += 1
                _upsert_existing_file_metadata(
                    existing_by_id=existing_by_id,
                    dataset_name=dataset_name,
                    dataset_index=int(idx),
                    patch_path=patch_path,
                    scale=scale,
                    patch_size_px=patch_size_px,
                    row=row,
                )
                continue

            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                target_iso = parse_date_iso(row["Sample Date"])
            except Exception:
                print(f"    [{idx}] invalid row (lat/lon/date parse failed)")
                continue

            image, region, metadata = find_nearest_modis_image(
                lat=lat,
                lon=lon,
                target_date_iso=target_iso,
                patch_size_px=patch_size_px,
                scale=scale,
                max_days_search=max_days_search,
            )
            prepared.append(
                {
                    "dataset_index": int(idx),
                    "lat": lat,
                    "lon": lon,
                    "target_date_iso": target_iso,
                    "image": image,
                    "region": region,
                }
            )
            meta_query[f"sample_{idx}"] = metadata

        if not meta_query:
            continue

        all_meta = ee.Dictionary(meta_query).getInfo()
        found_batch = 0
        for item in prepared:
            idx = item["dataset_index"]
            meta = all_meta.get(f"sample_{idx}", {})
            total_processed += 1
            if not meta.get("has_images", False):
                continue
            found_batch += 1
            total_found += 1
            patch_name = f"patch_{idx}"
            file_path = os.path.join(dataset_dir, f"{patch_name}.tif")
            samples_to_download.append(
                {
                    "dataset": dataset_name,
                    "dataset_index": idx,
                    "patch_name": patch_name,
                    "file_path": file_path,
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "target_date_iso": item["target_date_iso"],
                    "actual_date": meta.get("actual_date"),
                    "days_diff": meta.get("days_diff"),
                    "candidate_count": meta.get("candidate_count"),
                    "search_start": meta.get("search_start"),
                    "search_end_exclusive": meta.get("search_end_exclusive"),
                    "image": item["image"],
                    "region": item["region"],
                }
            )
            if idx % 20 == 0 or found_batch <= 5:
                print(
                    f"    [{idx}] found {meta.get('actual_date')} "
                    f"({meta.get('days_diff')}d away)"
                )

        rate = (found_batch / len(prepared) * 100) if prepared else 0.0
        print(
            f"    Batch found: {found_batch}/{len(prepared)} ({rate:.1f}%) | "
            f"Total found: {total_found}/{total_processed}"
        )

    print(f"\n  Candidates for download: {len(samples_to_download)}")
    print(f"  Resumed existing local files: {resumed_existing}")
    if not samples_to_download:
        print("  No samples found in search windows; nothing to download.")
        _save_metadata(metadata_json, metadata_csv, list(existing_by_id.values()))
        print(f"\n  Metadata updated:")
        print(f"    - {metadata_json}")
        print(f"    - {metadata_csv}")
        return

    print(f"  Starting downloads with {download_workers} workers...")
    total = len(samples_to_download)
    downloaded = 0
    skipped = 0
    failed = 0
    download_results = {}
    t0 = time.time()

    with cf.ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [
            executor.submit(
                _download_patch_local,
                sample,
                patch_size_px,
                scale,
                download_retries,
                initial_backoff_seconds,
            )
            for sample in samples_to_download
        ]
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            idx, ok, msg = fut.result()
            download_results[idx] = {"ok": ok, "message": msg}
            if ok:
                if msg == "exists":
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed += 1
            if i <= 20 or i % 100 == 0 or not ok:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(
                    f"    [{i}/{total}] patch_{idx}: {msg} | "
                    f"ok={downloaded} skip={skipped} fail={failed} | ETA {eta/60:.1f} min"
                )

    print("\n  Download summary")
    print(f"    Downloaded: {downloaded}")
    print(f"    Existing:   {skipped}")
    print(f"    Failed:     {failed}")

    for sample in samples_to_download:
        idx = int(sample["dataset_index"])
        result = download_results.get(idx, {"ok": False, "message": "not_attempted"})
        status = (
            "downloaded"
            if result["ok"] and result["message"] != "exists"
            else ("exists" if result["ok"] else "failed")
        )
        existing_by_id[idx] = {
            "dataset": sample["dataset"],
            "dataset_index": idx,
            "patch_id": idx,
            "file": sample["file_path"],
            "lat": float(sample["lat"]),
            "lon": float(sample["lon"]),
            "target_date_iso": sample["target_date_iso"],
            "target_date": datetime.strptime(sample["target_date_iso"], "%Y-%m-%d").strftime(
                "%d-%m-%Y"
            ),
            "actual_date": sample["actual_date"],
            "days_difference": float(sample["days_diff"]),
            "modis_scale_meters": int(scale),
            "patch_size_px": int(patch_size_px),
            "bands": OUT_BANDS,
            "qa_policy": "band_quality<=1 for bands [4,1,2,6,7]",
            "candidate_count": int(sample["candidate_count"]),
            "search_start": sample["search_start"],
            "search_end_exclusive": sample["search_end_exclusive"],
            "download_status": status,
            "download_message": result["message"],
        }

    _save_metadata(metadata_json, metadata_csv, list(existing_by_id.values()))
    print(f"\n  Metadata updated:")
    print(f"    - {metadata_json}")
    print(f"    - {metadata_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MODIS raw reflectance patches (CNN-ready) for location/date rows."
    )
    parser.add_argument("--training-csv", default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--submission-csv", default=DEFAULT_SUBMISSION_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", choices=["training", "submission", "both"], default="both")
    parser.add_argument("--ee-project", default=DEFAULT_EE_PROJECT)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--patch-size-px", type=int, default=DEFAULT_PATCH_SIZE_PX)
    parser.add_argument("--max-days-search", type=int, default=DEFAULT_MAX_DAYS_SEARCH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument(
        "--initial-backoff-seconds", type=float, default=DEFAULT_INITIAL_BACKOFF_SECONDS
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--use-high-volume-endpoint",
        action="store_true",
        help="Use Earth Engine high-volume endpoint for local getDownloadURL workload.",
    )
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if args.patch_size_px <= 0:
        raise ValueError("--patch-size-px must be > 0")
    if args.max_days_search < 0:
        raise ValueError("--max-days-search must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.download_workers <= 0:
        raise ValueError("--download-workers must be > 0")
    if args.download_retries <= 0:
        raise ValueError("--download-retries must be > 0")

    initialize_ee(args.ee_project, args.use_high_volume_endpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in {"training", "both"}:
        process_dataset(
            dataset_name="training",
            csv_path=args.training_csv,
            output_dir=args.output_dir,
            scale=args.scale,
            patch_size_px=args.patch_size_px,
            max_days_search=args.max_days_search,
            batch_size=args.batch_size,
            download_workers=args.download_workers,
            download_retries=args.download_retries,
            initial_backoff_seconds=args.initial_backoff_seconds,
            start_index=args.start_index,
            num_samples=args.num_samples,
        )

    if args.dataset in {"submission", "both"}:
        process_dataset(
            dataset_name="submission",
            csv_path=args.submission_csv,
            output_dir=args.output_dir,
            scale=args.scale,
            patch_size_px=args.patch_size_px,
            max_days_search=args.max_days_search,
            batch_size=args.batch_size,
            download_workers=args.download_workers,
            download_retries=args.download_retries,
            initial_backoff_seconds=args.initial_backoff_seconds,
            start_index=args.start_index,
            num_samples=args.num_samples,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
