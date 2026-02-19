import ee
import pandas as pd
from datetime import datetime
import os
import time
import urllib.request
import concurrent.futures as cf
import json
import argparse

# ---------------------------
# Parameters
# ---------------------------

CSV_PATH = "water_quality_training_dataset.csv"
START_INDEX = 0  # Set e.g. 600 to process rows 600..(600+NUM_SAMPLES-1)
NUM_SAMPLES = None  # None = process all rows
PATCH_SIZE = 256
SCALE = 30
HALF_SIZE_METERS = (PATCH_SIZE * SCALE) / 2

MAX_CLOUD_COVER = 100  # Accept any cloud cover
# Expand search window significantly to find any available imagery
MAX_DAYS_SEARCH = 2000  # Search up to 2 years before/after for difficult locations

METADATA_BATCH_SIZE = 100  # Smaller metadata batches reduce memory spikes on long runs

# Local download settings (no Drive / Cloud Storage exports)
MAX_DOWNLOAD_WORKERS = 8
MAX_DOWNLOAD_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.5
DOWNLOAD_FORMAT = "GEO_TIFF"  # single multiband tif per sample

# Use the high-volume endpoint for many small interactive downloads.
# Best for local getDownloadURL workflows.
USE_HIGH_VOLUME_ENDPOINT = True
if USE_HIGH_VOLUME_ENDPOINT:
    ee.Initialize(
        project='gearth-486818',
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
else:
    ee.Initialize(project='gearth-486818')
LANDSAT_5 = "LANDSAT/LT05/C02/T1_L2"
LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"

# Green needed for MNDWI; order: Green, Red, NIR, SWIR1, SWIR2
BANDS_L5 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
BANDS_L89 = ["SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

OUTPUT_FOLDER = "river_patches"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
METADATA_JSON = os.path.join(OUTPUT_FOLDER, "metadata.json")
METADATA_CSV = os.path.join(OUTPUT_FOLDER, "metadata.csv")

# ---------------------------
# Helper functions (keeping your existing ones)
# ---------------------------

def get_patch_geometry(lat, lon):
    """Create a square patch geometry around a point."""
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(HALF_SIZE_METERS).bounds()

def mask_scale_landsat_5(image):
    """Mask clouds and scale Landsat 5 SR data (SR bands only)."""
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    optical_bands = image.select(BANDS_L5).multiply(0.0000275).add(-0.2).toFloat()
    
    # Preserve temporal/source properties needed downstream (.date(), cloud metadata, etc.)
    return (
        optical_bands.updateMask(qa_mask)
        .copyProperties(image, image.propertyNames())
    )

def mask_scale_landsat_89(image):
    """Mask clouds and scale Landsat 8/9 SR data (SR bands only)."""
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    optical_bands = image.select(BANDS_L89).multiply(0.0000275).add(-0.2).toFloat()
    
    # Preserve temporal/source properties needed downstream (.date(), cloud metadata, etc.)
    return (
        optical_bands.updateMask(qa_mask)
        .copyProperties(image, image.propertyNames())
    )

def find_closest_image_optimized(lat, lon, target_date, region, max_days=None):
    """Find the nearest-date Landsat image, cloud as tie-breaker. Uses progressive relaxation if no match."""
    if max_days is None:
        max_days = MAX_DAYS_SEARCH
    
    target = ee.Date(target_date)
    start = target.advance(-max_days, 'day')
    end = target.advance(max_days, 'day')
    ls8_start = ee.Date('2013-03-18')
    
    # No cloud cover filtering - accept any imagery
    landsat_5_coll = (
        ee.ImageCollection(LANDSAT_5)
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_scale_landsat_5)
    )
    
    landsat_8_coll = (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_scale_landsat_89)
    )
    
    # Avoid querying L8 when the sampling date predates L8 availability.
    # This reduces unnecessary search load for 2011-early 2013 samples.
    collection = ee.ImageCollection(
        ee.Algorithms.If(target.millis().lt(ls8_start.millis()), landsat_5_coll, landsat_5_coll.merge(landsat_8_coll))
    )
    collection = collection.map(
        lambda img: img.set({
            'days_diff': ee.Number(img.date().difference(target, 'day')).abs(),
            # Days difference dominates ranking, cloud is tie-breaker.
            'rank_score': ee.Number(img.date().difference(target, 'day')).abs().multiply(1000)
                .add(ee.Number(img.get('CLOUD_COVER')))
        })
    )
    
    has_images = collection.size().gt(0)
    sorted_coll = collection.sort("rank_score")
    
    # Create dummy image with required properties set
    dummy_image = ee.Image.constant(0).set({
        'system:time_start': target.millis(),
        'CLOUD_COVER': 100,
        'SPACECRAFT_ID': 'NONE'
    })
    
    # Get best image or dummy
    best_image = ee.Image(
        ee.Algorithms.If(has_images, sorted_coll.first(), dummy_image)
    )
    
    # Now we can safely call .date() and .get() because both paths have the properties
    best_date = ee.Algorithms.If(
        has_images,
        best_image.date().format('YYYY-MM-dd'),
        'NO_IMAGE'
    )
    
    days_diff = ee.Algorithms.If(
        has_images,
        best_image.date().difference(target, 'day').abs(),
        -1
    )
    
    cloud_cover = ee.Algorithms.If(
        has_images,
        best_image.get('CLOUD_COVER'),
        -1
    )
    
    satellite_id = ee.Algorithms.If(
        has_images,
        best_image.get('SPACECRAFT_ID'),
        'NONE'
    )
    
    metadata_dict = ee.Dictionary({
        'has_images': has_images,
        'date': best_date,
        'days_diff': days_diff,
        'cloud_cover': cloud_cover,
        'satellite': satellite_id,
        'max_days_search': max_days
    })

    return best_image, metadata_dict


def load_existing_metadata():
    """Load existing metadata keyed by patch_id (dataset index)."""
    if not os.path.exists(METADATA_JSON):
        return {}
    try:
        with open(METADATA_JSON, "r") as f:
            existing = json.load(f)
        return {int(item["patch_id"]): item for item in existing}
    except Exception:
        return {}


def save_metadata(metadata_records):
    """Save metadata to JSON and CSV in output folder."""
    records_sorted = sorted(metadata_records, key=lambda x: int(x["patch_id"]))
    with open(METADATA_JSON, "w") as f:
        json.dump(records_sorted, f, indent=2)
    pd.DataFrame(records_sorted).to_csv(METADATA_CSV, index=False)

# ---------------------------
# Main Pipeline
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Landsat patches for water quality locations")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with locations (default: water_quality_training_dataset.csv)")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--num", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--output", type=str, default="river_patches", help="Output folder (default: river_patches)")
    args = parser.parse_args()
    
    if args.csv:
        CSV_PATH = args.csv
    if args.start:
        START_INDEX = args.start
    if args.num:
        NUM_SAMPLES = args.num
    if args.output:
        OUTPUT_FOLDER = args.output
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        METADATA_JSON = os.path.join(OUTPUT_FOLDER, "metadata.json")
        METADATA_CSV = os.path.join(OUTPUT_FOLDER, "metadata.csv")

print("="*80)
print("🚀 BATCHED LANDSAT PATCH EXTRACTION PIPELINE")
print("="*80)

df = pd.read_csv(CSV_PATH)
if NUM_SAMPLES is None:
    end_index = len(df)
else:
    end_index = min(START_INDEX + NUM_SAMPLES, len(df))
df_subset = df.iloc[START_INDEX:end_index]

print(f"\nProcessing {len(df_subset)} samples...")
print(f"  Dataset row range: {START_INDEX}..{end_index - 1}")
print(f"  Local download workers: {MAX_DOWNLOAD_WORKERS}")
print(f"  Download mode: direct getDownloadURL (no batch export tasks)")
print("  Cloud processing: SR bands + cloud mask only (NDVI/NDMI/water mask computed locally)")

# PHASE 1: Find images and prepare patches
print(f"\n🔍 PHASE 1: Finding images in batches of {METADATA_BATCH_SIZE}...")
samples_to_process = []
total_processed = 0
total_found = 0
num_batches = (len(df_subset) + METADATA_BATCH_SIZE - 1) // METADATA_BATCH_SIZE

for batch_num in range(num_batches):
    start_idx = batch_num * METADATA_BATCH_SIZE
    end_idx = min(start_idx + METADATA_BATCH_SIZE, len(df_subset))
    batch_df = df_subset.iloc[start_idx:end_idx]
    
    print(f"\n  📦 Batch {batch_num + 1}/{num_batches}: Samples {start_idx}-{end_idx-1}")
    
    samples_prep = []
    batch_metadata_dict = {}
    
    for idx, row in batch_df.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        date_str = row["Sample Date"]
        date = datetime.strptime(date_str, "%d-%m-%Y")
        target_date = date.strftime("%Y-%m-%d")
        region = get_patch_geometry(lat, lon)
        
        image, metadata_dict = find_closest_image_optimized(
            lat, lon, target_date, region
        )
        
        samples_prep.append({
            'idx': idx,
            'lat': lat,
            'lon': lon,
            'date_str': date_str,
            'target_date': target_date,
            'region': region,
            'image': image,
            'metadata': metadata_dict
        })
        
        batch_metadata_dict[f'sample_{idx}'] = metadata_dict
    
    print(f"     Fetching metadata for {len(samples_prep)} samples...")
    all_metadata = ee.Dictionary(batch_metadata_dict).getInfo()
    
    batch_found = 0
    for sample in samples_prep:
        idx = sample['idx']
        meta = all_metadata[f'sample_{idx}']
        total_processed += 1
        
        if not meta['has_images']:
            search_days = meta.get('max_days_search', MAX_DAYS_SEARCH)
            print(f"     [{idx}] ❌ No image found (lat={sample['lat']:.4f}, lon={sample['lon']:.4f}, date={sample['target_date']}, search_window=±{search_days}d)")
            continue
        
        batch_found += 1
        total_found += 1
        
        satellite = str(meta.get('satellite', '')).upper()
        bands = BANDS_L5 if "LANDSAT_5" in satellite else BANDS_L89
        # Export SR bands only; force float32 consistency across all bands.
        patch = sample['image'].select(bands).toFloat().clip(sample['region'])
        
        samples_to_process.append({
            'idx': idx,
            'lat': sample['lat'],
            'lon': sample['lon'],
            'target_date': sample['target_date'],
            'actual_date': meta['date'],
            'days_diff': meta['days_diff'],
            'cloud_cover': meta['cloud_cover'],
            'satellite': meta.get('satellite', 'UNKNOWN'),
            'region': sample['region'],
            'patch': patch
        })
        
        if idx % 10 == 0 or len(samples_to_process) <= 5:
            print(f"     [{idx}] ✓ {meta['date']} ({meta['days_diff']:.0f}d away, ☁️ {meta['cloud_cover']:.1f}%)")
    
    success_rate = (batch_found / len(samples_prep) * 100) if samples_prep else 0
    print(f"     ✓ Batch: {batch_found}/{len(samples_prep)} ({success_rate:.1f}%) | Total: {total_found}/{total_processed} ({total_found/total_processed*100:.1f}%)")

print(f"\n✅ Found {len(samples_to_process)} valid samples")

def download_patch_local(sample):
    """Download one patch directly to local disk using getDownloadURL + HTTP GET."""
    idx = sample['idx']
    out_path = os.path.join(OUTPUT_FOLDER, f"patch_{idx}.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return idx, True, "exists"

    params = {
        "name": f"patch_{idx}",
        "region": sample['region'],
        "scale": SCALE,
        "filePerBand": False,
        "format": DOWNLOAD_FORMAT,
    }

    backoff = INITIAL_BACKOFF_SECONDS
    last_err = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            url = sample['patch'].getDownloadURL(params)
            req = urllib.request.Request(url, headers={"User-Agent": "watah-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                content = resp.read()
            with open(out_path, "wb") as f:
                f.write(content)
            return idx, True, f"downloaded(attempt={attempt})"
        except Exception as exc:
            last_err = str(exc)
            # Exponential backoff with a cap
            time.sleep(min(backoff, 30))
            backoff *= 2

    return idx, False, f"failed: {last_err}"


# PHASE 2: Direct local downloads (parallel interactive requests)
print(f"\n⬇️  PHASE 2: Local direct downloads (no Drive / no Cloud Storage)...")
print(f"   Workers: {MAX_DOWNLOAD_WORKERS} | Retries: {MAX_DOWNLOAD_RETRIES}")
print(f"   Output folder: {OUTPUT_FOLDER}")

total = len(samples_to_process)
downloaded = 0
skipped = 0
failed = 0
start_time = time.time()
download_results = {}

with cf.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
    futures = [executor.submit(download_patch_local, sample) for sample in samples_to_process]
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
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"   [{i}/{total}] patch_{idx}: {msg} | "
                f"ok={downloaded} skip={skipped} fail={failed} | "
                f"ETA {eta/60:.1f} min"
            )

elapsed = time.time() - start_time
print("\n" + "=" * 80)
print("✅ LOCAL DOWNLOAD PHASE COMPLETE")
print(f"   Downloaded: {downloaded}")
print(f"   Skipped existing: {skipped}")
print(f"   Failed: {failed}")
print(f"   Total processed: {total}")
print(f"   Elapsed: {elapsed/60:.1f} minutes")
print("=" * 80)

# Build and save metadata (merge with existing so reruns are resumable)
existing_by_id = load_existing_metadata()
for sample in samples_to_process:
    idx = int(sample["idx"])
    result = download_results.get(idx, {"ok": False, "message": "not_attempted"})
    status = "downloaded" if result["ok"] and result["message"] != "exists" else (
        "exists" if result["ok"] else "failed"
    )
    satellite = str(sample.get("satellite", "")).upper()
    bands = BANDS_L5 if "LANDSAT_5" in satellite else BANDS_L89

    existing_by_id[idx] = {
        "patch_id": idx,
        "dataset_index": idx,
        "file": os.path.join(OUTPUT_FOLDER, f"patch_{idx}.tif"),
        "lat": float(sample["lat"]),
        "lon": float(sample["lon"]),
        "target_date": datetime.strptime(sample["target_date"], "%Y-%m-%d").strftime("%d-%m-%Y"),
        "target_date_iso": sample["target_date"],
        "actual_date": sample["actual_date"],
        "days_difference": float(sample["days_diff"]),
        "cloud_cover": float(sample["cloud_cover"]),
        "satellite": sample.get("satellite", "UNKNOWN"),
        "bands": bands,  # SR band names in source collection
        "derived_local": ["NDVI", "NDMI", "water_mask", "MNDWI"],
        "download_status": status,
        "download_message": result["message"],
    }

save_metadata(list(existing_by_id.values()))
print(f"\n💾 Metadata updated:")
print(f"   - {METADATA_JSON}")
print(f"   - {METADATA_CSV}")
