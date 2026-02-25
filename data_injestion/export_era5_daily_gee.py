#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd


def _read_points_and_dates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Latitude", "Longitude", "Sample Date"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df["Latitude"], errors="coerce"),
            "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
            "date": pd.to_datetime(df["Sample Date"], format="%d-%m-%Y", errors="coerce"),
        }
    ).dropna()
    out["lat"] = out["lat"].round(6)
    out["lon"] = out["lon"].round(6)
    return out


def _build_points_and_dates(train_csv: Path, submission_csv: Path) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    a = _read_points_and_dates(train_csv)
    b = _read_points_and_dates(submission_csv)
    all_rows = pd.concat([a, b], ignore_index=True).dropna()
    if all_rows.empty:
        raise ValueError("No valid lat/lon/date rows found.")

    points = all_rows[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    d0 = all_rows["date"].min()
    d1 = all_rows["date"].max()
    return points, d0, d1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export daily ERA5 values for all unique train/submission points via "
            "Google Earth Engine in a single task."
        )
    )
    ap.add_argument("--train-csv", type=Path, default=Path("data/water_quality_training_dataset.csv"))
    ap.add_argument("--submission-csv", type=Path, default=Path("data/submission_template.csv"))
    ap.add_argument("--collection", default="ECMWF/ERA5/DAILY")
    ap.add_argument(
        "--bands",
        default="",
        help="Comma-separated band names. Leave empty to export all bands.",
    )
    ap.add_argument("--scale", type=float, default=27830.0, help="Sampling scale in meters.")
    ap.add_argument("--drive-folder", default="awqp_era5")
    ap.add_argument("--file-prefix", default="era5_daily_awqp_points")
    ap.add_argument("--ee-project", default="", help="Optional Earth Engine project id.")
    ap.add_argument("--limit-points", type=int, default=0, help="If >0, export only first N points (debug).")
    args = ap.parse_args()

    points, d0, d1 = _build_points_and_dates(args.train_csv, args.submission_csv)
    if args.limit_points > 0:
        points = points.iloc[: args.limit_points].copy()

    print(f"Unique points: {len(points)}")
    print(f"Date range: {d0.date().isoformat()} .. {d1.date().isoformat()}")
    print(f"Collection: {args.collection}")

    # Lazy import because some environments don't have ee initialized yet.
    import ee  # type: ignore

    try:
        if args.ee_project:
            ee.Initialize(project=args.ee_project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if args.ee_project:
            ee.Initialize(project=args.ee_project)
        else:
            ee.Initialize()

    features = []
    for i, row in points.reset_index(drop=True).iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        geom = ee.Geometry.Point([lon, lat])
        features.append(
            ee.Feature(geom, {"point_id": int(i), "lat": lat, "lon": lon})
        )
    points_fc = ee.FeatureCollection(features)

    # filterDate end is exclusive in EE.
    end_exclusive = (d1 + timedelta(days=1)).date().isoformat()
    start_inclusive = d0.date().isoformat()

    ic = ee.ImageCollection(args.collection).filterDate(start_inclusive, end_exclusive)
    if args.bands.strip():
        bands = [b.strip() for b in args.bands.split(",") if b.strip()]
        ic = ic.select(bands)

    def sample_one_day(img):
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        sampled = img.sampleRegions(
            collection=points_fc,
            scale=args.scale,
            geometries=False,
        )
        return sampled.map(lambda f: f.set("date", date_str))

    out_fc = ee.FeatureCollection(ic.map(sample_one_day).flatten())
    n_images = ic.size().getInfo()
    print(f"Daily images selected: {n_images}")
    print(f"Estimated rows ~= {n_images} x {len(points)} = {n_images * len(points)}")

    task = ee.batch.Export.table.toDrive(
        collection=out_fc,
        description=args.file_prefix,
        folder=args.drive_folder,
        fileNamePrefix=args.file_prefix,
        fileFormat="CSV",
    )
    task.start()

    print("\nExport task started.")
    print(f"Description: {args.file_prefix}")
    print(f"Drive folder: {args.drive_folder}")
    print("Check task status in Earth Engine Tasks (Code Editor) or via ee.batch.Task.list().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
