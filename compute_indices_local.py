import argparse
import os
from glob import glob

import numpy as np
import rasterio


def compute_indices(input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        if src.count < 4:
            raise ValueError(f"{input_tif} has {src.count} band(s); expected at least 4")

        red = src.read(1).astype(np.float32)
        nir = src.read(2).astype(np.float32)
        swir1 = src.read(3).astype(np.float32)
        swir2 = src.read(4).astype(np.float32)

        ndvi = (nir - red) / np.where((nir + red) == 0, np.nan, (nir + red))
        ndmi = (nir - swir1) / np.where((nir + swir1) == 0, np.nan, (nir + swir1))
        water_mask = ((ndvi < 0.1) & (ndmi < 0)).astype(np.float32)

        # Keep NoData as NaN-compatible float32
        ndvi = np.nan_to_num(ndvi, nan=0.0).astype(np.float32)
        ndmi = np.nan_to_num(ndmi, nan=0.0).astype(np.float32)

        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=7, compress="deflate")

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(red, 1)
            dst.write(nir, 2)
            dst.write(swir1, 3)
            dst.write(swir2, 4)
            dst.write(water_mask, 5)
            dst.write(ndvi, 6)
            dst.write(ndmi, 7)


def main():
    parser = argparse.ArgumentParser(
        description="Compute water_mask, NDVI, NDMI for local Landsat patch tifs."
    )
    parser.add_argument("--input-folder", type=str, default="river_patches")
    parser.add_argument(
        "--output-folder",
        type=str,
        default="river_patches_with_indices",
        help="Folder for 7-band output tifs (bands 1-4 + water_mask + NDVI + NDMI).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="patch_*.tif",
        help="Glob pattern inside input folder.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    tif_paths = sorted(glob(os.path.join(args.input_folder, args.pattern)))

    print(f"Found {len(tif_paths)} tif file(s) in {args.input_folder}")
    done = 0
    skipped = 0
    failed = 0

    for i, tif_path in enumerate(tif_paths, start=1):
        out_path = os.path.join(args.output_folder, os.path.basename(tif_path))
        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            compute_indices(tif_path, out_path)
            done += 1
            if i <= 20 or i % 250 == 0:
                print(f"[{i}/{len(tif_paths)}] ✓ {os.path.basename(tif_path)}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(tif_paths)}] ✗ {os.path.basename(tif_path)}: {exc}")

    print("\nDone.")
    print(f"Computed: {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
