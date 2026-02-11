import ee
import json
import os
from datetime import datetime
import argparse

ee.Initialize(project='gearth-486818')

# ---------------------------
# Parameters
# ---------------------------

BATCH_SIZE = 50  # Process climate features in batches

# ---------------------------
# Climate Feature Functions
# ---------------------------

def get_rainfall_7day(lat, lon, date):
    """
    Get accumulated rainfall for 7 days BEFORE the sampling date.
    
    For a sample on date D, we want rainfall from [D-7, D-1] inclusive,
    i.e., the 7 days leading up to (but not including) the sampling day.
    """
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(5000)  # 5km buffer
    
    end_date = ee.Date(date)  # Sampling date (exclusive in filterDate)
    start_date = end_date.advance(-7, 'day')  # 7 days before
    
    # CHIRPS daily rainfall (1981-present, quasi-global)
    # filterDate is inclusive of start, exclusive of end
    # So this gives us [start_date, end_date) = 7 days before the sample
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .select('precipitation') \
        .sum()  # Total rainfall over 7 days
    
    rainfall_mean = rainfall.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=5566  # CHIRPS resolution ~5km
    ).get('precipitation')
    
    return rainfall_mean

def get_temperature(lat, lon, date):
    """
    Get temperature around the sampling date (±3 day window).
    Returns ee.Number for batching (Celsius).
    """
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(5000)
    
    sample_date = ee.Date(date)
    start = sample_date.advance(-3, 'day')
    end = sample_date.advance(4, 'day')
    
    # ERA5 daily temperature (1979-present, global)
    temp = ee.ImageCollection('ECMWF/ERA5/DAILY') \
        .filterDate(start, end) \
        .select('mean_2m_air_temperature') \
        .mean()
    
    temp_mean = temp.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=27830  # ERA5 resolution ~28km
    ).get('mean_2m_air_temperature')
    
    # Convert from Kelvin to Celsius (server-side)
    return ee.Number(temp_mean).subtract(273.15)

def main():
    parser = argparse.ArgumentParser(
        description="Add climate features to metadata entries for local patch tifs."
    )
    parser.add_argument("--patch-folder", type=str, default="river_patches")
    parser.add_argument("--metadata-file", type=str, default="metadata.json")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="If omitted, overwrite metadata file in patch folder.",
    )
    parser.add_argument(
        "--only-downloaded",
        action="store_true",
        help="Only process entries with download_status in {'downloaded','exists'} and file present.",
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.patch_folder, args.metadata_file)
    output_path = (
        os.path.join(args.patch_folder, args.output_file)
        if args.output_file
        else metadata_path
    )

    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Optional filter: only patches that were actually downloaded.
    selected_indices = []
    for i, sample in enumerate(metadata):
        if not args.only_downloaded:
            selected_indices.append(i)
            continue
        status = sample.get("download_status", "")
        file_path = sample.get("file", "")
        if status in {"downloaded", "exists"} and os.path.exists(file_path):
            selected_indices.append(i)

    print(f"Found {len(metadata)} total entries")
    print(f"Processing {len(selected_indices)} entries for climate features")
    print(f"\n🌡️  Processing climate features in batches of {BATCH_SIZE}...\n")

    num_batches = (len(selected_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(selected_indices))
        batch_meta_indices = selected_indices[start_idx:end_idx]
        batch = [metadata[idx] for idx in batch_meta_indices]

        print(f"📦 Batch {batch_num + 1}/{num_batches}: Entries {start_idx}-{end_idx-1}")

        batch_dict = {}
        for i, sample in enumerate(batch):
            target_iso = sample.get("target_date_iso")
            if not target_iso:
                date_obj = datetime.strptime(sample["target_date"], "%d-%m-%Y")
                target_iso = date_obj.strftime("%Y-%m-%d")

            batch_dict[f"sample_{i}_rain"] = get_rainfall_7day(
                sample["lat"], sample["lon"], target_iso
            )
            batch_dict[f"sample_{i}_temp"] = get_temperature(
                sample["lat"], sample["lon"], target_iso
            )

        print("   Fetching climate data...")
        all_features = ee.Dictionary(batch_dict).getInfo()

        for i, sample in enumerate(batch):
            sample["scalar_features"] = {
                "rainfall_7day_mm": all_features[f"sample_{i}_rain"],
                "temperature_celsius": all_features[f"sample_{i}_temp"],
            }

            patch_id = sample.get("patch_id", i)
            rain = sample["scalar_features"]["rainfall_7day_mm"]
            temp = sample["scalar_features"]["temperature_celsius"]
            if i < 3 or patch_id % 100 == 0:
                print(f"   [{patch_id}] 🌧️ {rain:.1f}mm | 🌡️ {temp:.1f}°C")

            metadata_idx = batch_meta_indices[i]
            metadata[metadata_idx] = sample

    print(f"\n✅ Completed climate feature extraction for {len(selected_indices)} entries")
    print(f"💾 Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("✅ Done! Metadata updated with climate features.")


if __name__ == "__main__":
    main()
