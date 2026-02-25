#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import socket
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

import pandas as pd
from tqdm import tqdm


DEFAULT_COLLECTION = "ECMWF/ERA5/DAILY"
DEFAULT_LOOKBACK_DAYS = 30
MAX_DOWNLOAD_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.5


def _read_rows(csv_path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required = {"Latitude", "Longitude", "Sample Date"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["sample_date"] = pd.to_datetime(df["Sample Date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "sample_date"]).copy()
    df["source"] = source_name
    df["source_row"] = df.index.astype(int)
    df["lat_r6"] = df["Latitude"].round(6)
    df["lon_r6"] = df["Longitude"].round(6)
    return df


def _download_url_to_file(
    url: str,
    out_path: Path,
    *,
    timeout_seconds: int,
    max_retries: int,
) -> None:
    backoff = INITIAL_BACKOFF_SECONDS
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "awqp-era5/1.0"})
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(tmp_path, out_path)
            return
        except (TimeoutError, socket.timeout, URLError, OSError) as exc:  # pragma: no cover
            last_err = exc
            if attempt < max_retries:
                time.sleep(min(backoff, 30))
                backoff *= 2
    raise RuntimeError(f"Failed to download after retries: {last_err}")


def _build_daily_timeseries(
    *,
    points_df: pd.DataFrame,
    collection: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    scale_m: float,
    bands_csv: str,
    point_batch_size: int,
    out_dir: Path,
    timeout_seconds: int,
    max_retries: int,
    resume: bool,
) -> pd.DataFrame:
    import ee  # type: ignore

    start_iso = start_date.date().isoformat()
    end_exclusive_iso = (end_date + pd.Timedelta(days=1)).date().isoformat()

    ic_base = ee.ImageCollection(collection).filterDate(start_iso, end_exclusive_iso)
    if bands_csv.strip():
        bands = [b.strip() for b in bands_csv.split(",") if b.strip()]
        ic_base = ic_base.select(bands)

    n_days = int(ic_base.size().getInfo())
    n_points = len(points_df)
    n_batches = math.ceil(n_points / point_batch_size)
    print(f"Daily images selected: {n_days}")
    print(f"Points: {n_points} (batch size {point_batch_size} -> {n_batches} batches)")
    print(f"Estimated rows: {n_days} * {n_points} = {n_days * n_points}")

    temp_dir = out_dir / "_tmp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for batch_idx in tqdm(range(n_batches), desc="Downloading ERA5 point batches", unit="batch"):
        s = batch_idx * point_batch_size
        e = min(s + point_batch_size, n_points)
        batch_points = points_df.iloc[s:e]

        features = []
        for row in batch_points.itertuples(index=False):
            geom = ee.Geometry.Point([float(row.lon_r6), float(row.lat_r6)])
            feat = ee.Feature(
                geom,
                {
                    "point_id": int(row.point_id),
                    "lat": float(row.lat_r6),
                    "lon": float(row.lon_r6),
                },
            )
            features.append(feat)
        points_fc = ee.FeatureCollection(features)

        def sample_one_day(img):
            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            sampled = img.sampleRegions(
                collection=points_fc,
                scale=scale_m,
                geometries=False,
            )
            return sampled.map(lambda f: f.set("date", date_str))

        out_fc = ee.FeatureCollection(ic_base.map(sample_one_day).flatten())
        tmp_csv = temp_dir / f"era5_daily_points_batch_{batch_idx:03d}.csv"
        if resume and tmp_csv.exists() and tmp_csv.stat().st_size > 0:
            tqdm.write(f"Batch {batch_idx + 1}/{n_batches}: using cached {tmp_csv.name}")
        else:
            url = out_fc.getDownloadURL(filetype="CSV")
            tqdm.write(f"Batch {batch_idx + 1}/{n_batches}: downloading {tmp_csv.name}")
            _download_url_to_file(
                url,
                tmp_csv,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

        batch_df = pd.read_csv(tmp_csv)
        batch_df["date"] = pd.to_datetime(batch_df["date"], errors="coerce")
        batch_df = batch_df.dropna(subset=["date"]).copy()
        if "point_id" not in batch_df.columns:
            raise KeyError(f"Downloaded CSV missing required 'point_id' column: {tmp_csv}")
        batch_df["point_id"] = pd.to_numeric(batch_df["point_id"], errors="coerce").astype("Int64")
        batch_df = batch_df.dropna(subset=["point_id"]).copy()
        batch_df["point_id"] = batch_df["point_id"].astype(int)
        frames.append(batch_df)

    if not frames:
        return pd.DataFrame(columns=["point_id", "date"])
    return pd.concat(frames, ignore_index=True)


def _build_lookback_features(
    sample_rows: pd.DataFrame,
    daily_df: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    key_cols = {"point_id", "date", "lat", "lon", "longitude", "latitude", ".geo"}
    value_cols = [c for c in daily_df.columns if c not in key_cols]
    value_cols = [c for c in value_cols if c != "system:index"]
    if not value_cols:
        raise ValueError("No ERA5 value columns found in downloaded daily data.")

    base = sample_rows.copy()
    base["sample_date"] = pd.to_datetime(base["sample_date"])
    out = base.copy()

    slim = daily_df[["point_id", "date", *value_cols]].copy()
    slim["date"] = pd.to_datetime(slim["date"])

    # Lag 1..lookback_days means previous days only (no target day leakage).
    for lag in tqdm(range(1, lookback_days + 1), desc="Building lookback lags", unit="lag"):
        rhs = slim.copy()
        rhs["sample_date"] = rhs["date"] + pd.Timedelta(days=lag)
        rename_map = {col: f"{col}_lag{lag:02d}" for col in value_cols}
        rhs = rhs.rename(columns=rename_map)
        rhs = rhs.drop(columns=["date"])
        out = out.merge(rhs, on=["point_id", "sample_date"], how="left")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Local Earth Engine download for ERA5 daily weather and 30-day lookback "
            "features per train/submission row."
        )
    )
    ap.add_argument("--train-csv", type=Path, default=Path("data/water_quality_training_dataset.csv"))
    ap.add_argument("--submission-csv", type=Path, default=Path("data/submission_template.csv"))
    ap.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    ap.add_argument("--bands", type=str, default="", help="Comma-separated band list; empty means all bands.")
    ap.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    ap.add_argument(
        "--point-batch-size",
        type=int,
        default=30,
        help="Number of points per Earth Engine download batch.",
    )
    ap.add_argument("--scale", type=float, default=27830.0)
    ap.add_argument("--ee-project", type=str, default="gearth-486818")
    ap.add_argument(
        "--high-volume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Earth Engine high-volume endpoint.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/era5"))
    ap.add_argument(
        "--download-timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout for each batch download.",
    )
    ap.add_argument(
        "--max-download-retries",
        type=int,
        default=7,
        help="Retries for each batch download.",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse already-downloaded batch CSV files when present.",
    )
    args = ap.parse_args()

    if args.lookback_days < 1:
        raise ValueError("--lookback-days must be >= 1")
    if args.point_batch_size < 1:
        raise ValueError("--point-batch-size must be >= 1")
    if args.download_timeout_seconds < 10:
        raise ValueError("--download-timeout-seconds must be >= 10")
    if args.max_download_retries < 1:
        raise ValueError("--max-download-retries must be >= 1")

    train_df = _read_rows(args.train_csv, "train")
    sub_df = _read_rows(args.submission_csv, "submission")
    samples = pd.concat([train_df, sub_df], ignore_index=True)

    unique_points = (
        samples[["lat_r6", "lon_r6"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index(names="point_id")
    )
    samples = samples.merge(unique_points, on=["lat_r6", "lon_r6"], how="left")

    d_min = samples["sample_date"].min() - pd.Timedelta(days=args.lookback_days)
    d_max = samples["sample_date"].max()

    print(f"Rows: {len(samples)}")
    print(f"Unique locations: {len(unique_points)}")
    print(f"Date range for ERA5 query: {d_min.date().isoformat()} .. {d_max.date().isoformat()}")
    print(f"Lookback days: {args.lookback_days}")
    print(f"Collection: {args.collection}")
    print(f"Resume mode: {args.resume}")

    import ee  # type: ignore

    if args.high_volume:
        ee.Initialize(project=args.ee_project, opt_url="https://earthengine-highvolume.googleapis.com")
    else:
        ee.Initialize(project=args.ee_project)

    daily_df = _build_daily_timeseries(
        points_df=unique_points,
        collection=args.collection,
        start_date=d_min,
        end_date=d_max,
        scale_m=args.scale,
        bands_csv=args.bands,
        point_batch_size=args.point_batch_size,
        out_dir=args.out_dir,
        timeout_seconds=args.download_timeout_seconds,
        max_retries=args.max_download_retries,
        resume=args.resume,
    )

    print("Building lookback feature table...")
    features_df = _build_lookback_features(
        sample_rows=samples,
        daily_df=daily_df,
        lookback_days=args.lookback_days,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_map_path = args.out_dir / "era5_points_map.csv"
    raw_daily_path = args.out_dir / "era5_daily_points.csv"
    lookback_path = args.out_dir / f"era5_lookback_{args.lookback_days}d.csv"

    unique_points.to_csv(points_map_path, index=False)
    daily_df.to_csv(raw_daily_path, index=False)
    features_df.to_csv(lookback_path, index=False)

    print("\nDone.")
    print(f"- Points map: {points_map_path}")
    print(f"- Raw daily point data: {raw_daily_path}")
    print(f"- Lookback features: {lookback_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
