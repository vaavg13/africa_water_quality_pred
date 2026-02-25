import ee
import pandas as pd
from datetime import datetime
import os
import argparse

# ---------------------------
# Parameters
# ---------------------------

CSV_PATH = "water_quality_training_dataset.csv"
START_INDEX = 0
NUM_SAMPLES = None  # None = all rows
PATCH_SIZE = 256
SCALE = 30
HALF_SIZE_METERS = (PATCH_SIZE * SCALE) / 2

MAX_CLOUD_COVER = 100  # Accept any cloud cover
MAX_DAYS_FORWARD = 180  # Only look up to 6 months ahead of target date

METADATA_BATCH_SIZE = 100

OUTPUT_CSV = "landsat_availability.csv"

ee.Initialize(
    project='gearth-486818',
    opt_url='https://earthengine-highvolume.googleapis.com'
)

LANDSAT_5 = "LANDSAT/LT05/C02/T1_L2"
LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"

# ---------------------------
# Helpers
# ---------------------------

def get_patch_geometry(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(HALF_SIZE_METERS).bounds()


def check_availability(lat, lon, target_date, region):
    """
    Find the nearest Landsat image taken AFTER target_date, within 6 months.
    Returns an ee.Dictionary with availability metadata.
    """
    target = ee.Date(target_date)
    start = target
    end = target.advance(MAX_DAYS_FORWARD, 'day')
    ls8_start = ee.Date('2013-03-18')

    landsat_5_coll = (
        ee.ImageCollection(LANDSAT_5)
        .filterBounds(region)
        .filterDate(start, end)
    )

    landsat_8_coll = (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(region)
        .filterDate(start, end)
    )

    collection = ee.ImageCollection(
        ee.Algorithms.If(
            target.millis().lt(ls8_start.millis()),
            landsat_5_coll,
            landsat_5_coll.merge(landsat_8_coll)
        )
    )

    collection = collection.map(
        lambda img: img.set({
            'days_diff': img.date().difference(target, 'day'),
            'rank_score': img.date().difference(target, 'day')
                .multiply(1000)
                .add(ee.Number(img.get('CLOUD_COVER')))
        })
    )

    has_images = collection.size().gt(0)
    sorted_coll = collection.sort('rank_score')

    dummy = ee.Image.constant(0).set({
        'system:time_start': target.millis(),
        'CLOUD_COVER': 100,
        'SPACECRAFT_ID': 'NONE'
    })

    best = ee.Image(ee.Algorithms.If(has_images, sorted_coll.first(), dummy))

    best_date = ee.Algorithms.If(
        has_images,
        best.date().format('YYYY-MM-dd'),
        'NO_IMAGE'
    )
    days_diff = ee.Algorithms.If(
        has_images,
        best.date().difference(target, 'day'),
        -1
    )
    cloud_cover = ee.Algorithms.If(
        has_images,
        best.get('CLOUD_COVER'),
        -1
    )
    satellite = ee.Algorithms.If(
        has_images,
        best.get('SPACECRAFT_ID'),
        'NONE'
    )
    image_count = collection.size()

    return ee.Dictionary({
        'has_images': has_images,
        'image_count': image_count,
        'closest_date': best_date,
        'days_after_target': days_diff,
        'cloud_cover': cloud_cover,
        'satellite': satellite,
    })


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Landsat availability (forward 6 months) for water quality locations")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.csv:
        CSV_PATH = args.csv
    if args.start:
        START_INDEX = args.start
    if args.num:
        NUM_SAMPLES = args.num
    if args.output:
        OUTPUT_CSV = args.output

    print("=" * 80)
    print("LANDSAT AVAILABILITY CHECK  (forward window: 0 to +6 months)")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    end_index = len(df) if NUM_SAMPLES is None else min(START_INDEX + NUM_SAMPLES, len(df))
    df_subset = df.iloc[START_INDEX:end_index]

    print(f"\nRows: {START_INDEX}..{end_index - 1}  ({len(df_subset)} samples)")
    print(f"Batch size: {METADATA_BATCH_SIZE}")
    print(f"Search window: target_date to target_date +{MAX_DAYS_FORWARD}d (no lookback)\n")

    results = []
    num_batches = (len(df_subset) + METADATA_BATCH_SIZE - 1) // METADATA_BATCH_SIZE

    for batch_num in range(num_batches):
        b_start = batch_num * METADATA_BATCH_SIZE
        b_end = min(b_start + METADATA_BATCH_SIZE, len(df_subset))
        batch_df = df_subset.iloc[b_start:b_end]

        print(f"  Batch {batch_num + 1}/{num_batches}: rows {b_start}-{b_end - 1}")

        batch_dict = {}
        row_meta = {}

        for idx, row in batch_df.iterrows():
            lat = row["Latitude"]
            lon = row["Longitude"]
            date_str = row["Sample Date"]
            target_date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            region = get_patch_geometry(lat, lon)

            batch_dict[f"s_{idx}"] = check_availability(lat, lon, target_date, region)
            row_meta[idx] = {
                "lat": lat,
                "lon": lon,
                "target_date": target_date,
                "target_date_orig": date_str,
            }

        all_meta = ee.Dictionary(batch_dict).getInfo()

        found = 0
        for idx, base in row_meta.items():
            m = all_meta[f"s_{idx}"]
            has = bool(m["has_images"])
            if has:
                found += 1

            results.append({
                "dataset_index": idx,
                "lat": base["lat"],
                "lon": base["lon"],
                "target_date": base["target_date"],
                "target_date_orig": base["target_date_orig"],
                "has_image": has,
                "image_count": int(m["image_count"]) if has else 0,
                "closest_date": m["closest_date"],
                "days_after_target": round(float(m["days_after_target"]), 1) if has else None,
                "cloud_cover": round(float(m["cloud_cover"]), 1) if has else None,
                "satellite": m["satellite"],
            })

        total_so_far = len(results)
        found_so_far = sum(1 for r in results if r["has_image"])
        print(f"    Found {found}/{len(row_meta)} | Running total: {found_so_far}/{total_so_far} ({found_so_far/total_so_far*100:.1f}%)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total = len(results)
    available = sum(1 for r in results if r["has_image"])
    print("\n" + "=" * 80)
    print("DONE")
    print(f"  Total samples checked : {total}")
    print(f"  With image available  : {available}  ({available/total*100:.1f}%)")
    print(f"  No image found        : {total - available}")
    print(f"  Output saved to       : {OUTPUT_CSV}")
    print("=" * 80)
