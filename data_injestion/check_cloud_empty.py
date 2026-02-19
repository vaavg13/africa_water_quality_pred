#!/usr/bin/env python3
"""Check what percentage of TIFFs and pixels are empty due to cloud coverage."""

import argparse
import numpy as np
import rasterio
from pathlib import Path

MODIS_DIR = Path("data/modis_patches/training")
LANDSAT_DIR = Path("data/river_patches")


def count_empty_pixels(tif_path: Path, zero_threshold: float = 1e-10) -> tuple[int, int]:
    """Return (empty_pixels, total_pixels). Uses band 1 (mask applies to all bands)."""
    try:
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            total = band.size
            nodata = src.nodata
            empty_mask = ~np.isfinite(band)
            if nodata is not None:
                empty_mask |= np.isclose(band, nodata)
            empty_mask |= band <= zero_threshold
            return int(np.sum(empty_mask)), total
    except Exception:
        return 0, 0


def run_analysis(tif_dir: Path, label: str):
    tifs = sorted(tif_dir.glob("patch_*.tif"))
    if not tifs:
        print(f"No TIFFs found in {tif_dir}")
        return

    total_empty_px = 0
    total_px = 0
    fully_empty_ids = []

    for tif_path in tifs:
        empty_px, px = count_empty_pixels(tif_path)
        total_empty_px += empty_px
        total_px += px
        if px > 0 and empty_px == px:
            patch_id = int(tif_path.stem.replace("patch_", ""))
            fully_empty_ids.append(patch_id)

    pct_empty_px = 100.0 * total_empty_px / total_px if total_px > 0 else 0
    pct_empty_tifs = 100.0 * len(fully_empty_ids) / len(tifs)

    print()
    print("=" * 60)
    print(f"{label} cloud coverage analysis")
    print("=" * 60)
    print(f"Total TIFFs:           {len(tifs)}")
    print(f"Fully empty TIFFs:     {len(fully_empty_ids)} ({pct_empty_tifs:.1f}%)")
    print(f"Total pixels:          {total_px:,}")
    print(f"Empty pixels:          {total_empty_px:,}")
    print(f"Empty pixel %:         {pct_empty_px:.2f}%")
    if fully_empty_ids:
        n_show = min(30, len(fully_empty_ids))
        print(f"Fully empty IDs (first {n_show}): {fully_empty_ids[:n_show]}")


def main():
    parser = argparse.ArgumentParser(description="Check cloud/empty pixel percentage in patch TIFFs")
    parser.add_argument(
        "--sensor",
        choices=["modis", "landsat"],
        default="modis",
        help="Dataset: modis (data/modis_patches/training) or landsat (data/river_patches)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.sensor == "modis":
        tif_dir = root / MODIS_DIR
        label = "MODIS"
    else:
        tif_dir = root / LANDSAT_DIR
        label = "Landsat"

    run_analysis(tif_dir, label)


if __name__ == "__main__":
    main()
