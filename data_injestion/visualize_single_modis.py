#!/usr/bin/env python3
"""Visualize a single MODIS patch with optional HydroRIVERS overlay.

Works in native Sinusoidal CRS. Query point placed via CRS transform, not center assumption.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_indices_local import (
    _build_hydrorivers_weighted_channel,
    _effective_crs,
    _resolve_hydrorivers_layer,
)

DEFAULT_PATCH_ID = 3
PATCHES_FOLDER = "data/modis_patches/training"
METADATA_PATH = "data/modis_patches/training/metadata.json"


def _normalize_for_display(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32).copy()
    band[~np.isfinite(band)] = np.nan
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return band
    p2, p98 = np.nanpercentile(band, (2, 98))
    if p98 == p2:
        return band
    normed = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
    normed[~np.isfinite(band)] = np.nan
    return normed.astype(np.float32)


def _make_cmap(name: str):
    cmap = plt.cm.get_cmap(name).copy()
    cmap.set_bad(color="lightgray")
    return cmap


def _resolve_tif_path(patch_id, patch_meta, patches_folder):
    basename = f"patch_{patch_id}.tif"
    candidate = os.path.join(patches_folder, basename)
    if os.path.exists(candidate):
        return candidate
    if patch_meta is not None and "file" in patch_meta:
        meta_file = patch_meta["file"]
        candidate = os.path.join(patches_folder, os.path.basename(meta_file))
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(meta_file):
            return meta_file
    raise FileNotFoundError(f"Could not find TIFF for patch_id={patch_id} in {patches_folder}")


def _lonlat_to_pixel(src, lon, lat):
    """Transform (lon, lat) WGS84 → pixel (row, col) using the effective CRS."""
    crs = _effective_crs(src)
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    row, col = src.index(xs[0], ys[0])
    row = max(0, min(row, src.height - 1))
    col = max(0, min(col, src.width - 1))
    return row, col


def visualize_single_modis(
    patch_id,
    patches_folder=None,
    metadata_path=None,
    hydrorivers_gdb_path=None,
    hydrorivers_layer=None,
    min_discharge_cms=0.0,
    min_strahler_order=1,
    discharge_cap_cms=1000.0,
    river_dilate_pixels=1,
):
    if patches_folder is None:
        patches_folder = PATCHES_FOLDER
    if metadata_path is None:
        metadata_path = METADATA_PATH

    patch_meta = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)
        for meta in all_metadata:
            if int(meta.get("patch_id", meta.get("dataset_index", -1))) == int(patch_id):
                patch_meta = meta
                break

    tif_path = _resolve_tif_path(patch_id, patch_meta, patches_folder)

    query_lat = patch_meta.get("lat") if patch_meta else None
    query_lon = patch_meta.get("lon") if patch_meta else None
    target_date = patch_meta.get("target_date") if patch_meta else None
    actual_date = patch_meta.get("actual_date") if patch_meta else None
    days_diff = patch_meta.get("days_difference") if patch_meta else None
    modis_scale = patch_meta.get("modis_scale_meters") if patch_meta else None
    satellite = f"MODIS ({modis_scale} m)" if modis_scale else "MODIS"

    print(f"Visualizing MODIS patch {patch_id}")
    print(f"  TIFF: {tif_path}")
    if query_lat is not None and query_lon is not None:
        print(f"  Location: ({query_lat}, {query_lon})")
    if target_date is not None:
        print(f"  Target date: {target_date}")
    if actual_date is not None and days_diff is not None:
        print(f"  Actual date: {actual_date} ({float(days_diff):.0f} days from target)")

    with rasterio.open(tif_path) as src:
        num_bands = src.count
        height, width = src.height, src.width

        if num_bands == 5:
            green = src.read(1).astype(np.float32)
            red = src.read(2).astype(np.float32)
            nir = src.read(3).astype(np.float32)
            swir1 = src.read(4).astype(np.float32)
            swir2 = src.read(5).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                for arr in (green, red, nir, swir1, swir2):
                    arr[np.isclose(arr, nodata)] = np.nan
        else:
            red = src.read(1).astype(np.float32)
            nir = src.read(2).astype(np.float32)
            swir1 = src.read(3).astype(np.float32)
            swir2 = src.read(4).astype(np.float32)
            green = None

        # Place query point via CRS transform (not center assumption)
        if query_lat is not None and query_lon is not None:
            row, col = _lonlat_to_pixel(src, query_lon, query_lat)
        else:
            row, col = height // 2, width // 2

        # HydroRIVERS in native Sinusoidal CRS
        hydro_weight = None
        if hydrorivers_gdb_path:
            print(f"  Computing HydroRIVERS (native CRS, dilate={river_dilate_pixels}px)…")
            resolved_layer = _resolve_hydrorivers_layer(
                hydrorivers_gdb_path, hydrorivers_layer
            )
            where_clause = (
                f"DIS_AV_CMS >= {float(min_discharge_cms)} "
                f"OR ORD_STRA >= {int(min_strahler_order)}"
            )
            hydro_weight = _build_hydrorivers_weighted_channel(
                src=src,
                gdb_path=hydrorivers_gdb_path,
                layer_name=resolved_layer,
                where_clause=where_clause,
                discharge_cap_cms=discharge_cap_cms,
                dilate_pixels=int(river_dilate_pixels),
            )
            hydro_weight = np.where(hydro_weight > 0, hydro_weight, np.nan).astype(
                np.float32
            )

    def _safe_ratio(a, b):
        denom = np.where(b == 0, np.nan, b)
        return (a / denom).astype(np.float32)

    ndvi = _safe_ratio(nir - red, nir + red)
    ndmi = _safe_ratio(nir - swir1, nir + swir1)
    water_mask = (
        np.where(np.isfinite(ndvi) & np.isfinite(ndmi), (ndvi < 0.1) & (ndmi < 0), np.nan)
    ).astype(np.float32)
    mndwi = (
        _safe_ratio(green - swir1, green + swir1)
        if green is not None
        else np.full_like(red, np.nan, dtype=np.float32)
    )

    valid_pct = float(np.isfinite(red).mean() * 100)
    print(f"  Query point pixel: row={row}, col={col}")
    print(f"  Image shape: {red.shape}")
    print(f"  Valid pixels: {valid_pct:.1f}%")
    if valid_pct < 5.0:
        print("  WARNING: >95% masked; image will appear mostly gray.")

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    bands_data = [
        (_normalize_for_display(red), "Red", "Reds"),
        (_normalize_for_display(nir), "NIR", "Greens"),
        (_normalize_for_display(swir1), "SWIR1", "YlOrBr"),
        (_normalize_for_display(swir2), "SWIR2", "copper"),
        (water_mask, "Water mask", "Blues"),
        (ndvi, "NDVI", "RdYlGn"),
        (ndmi, "NDMI", "BrBG"),
        (mndwi, "MNDWI", "PuBuGn"),
    ]
    if hydro_weight is not None:
        bands_data.append((hydro_weight, "HydroRIVERS weighted", "viridis"))

    for i, (band_data, title, cmap_name) in enumerate(bands_data):
        ax = axes[i]
        cmap = _make_cmap(cmap_name)
        im = ax.imshow(band_data, cmap=cmap)
        ax.plot(col, row, "r*", markersize=18, markeredgecolor="yellow", markeredgewidth=2)
        ax.axhline(y=row, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(x=col, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if len(bands_data) < len(axes):
        info_ax = axes[len(bands_data)]
        info_ax.axis("off")
        extra = f"Days from target: {float(days_diff):.0f}" if days_diff is not None else "N/A"
        info_ax.text(
            0.05,
            0.8,
            f"Patch: {patch_id}\nSensor: {satellite}\n{extra}\n"
            f"Valid pixels: {valid_pct:.1f}%\nBands: {num_bands}\n"
            f"Point pixel: ({row}, {col})",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for i in range(len(bands_data) + 1, len(axes)):
        axes[i].axis("off")

    title_text = f"Patch {patch_id} - {satellite}"
    if query_lat is not None and query_lon is not None:
        title_text += f"\nLat: {query_lat:.4f}, Lon: {query_lon:.4f}"
    if target_date is not None:
        title_text += f"\nTarget: {target_date}"
        if actual_date is not None:
            title_text += f", Actual: {actual_date}"
            if days_diff is not None:
                title_text += f" ({float(days_diff):.0f}d)"

    fig.suptitle(title_text, fontsize=14, y=0.99)
    plt.tight_layout()

    output_path = f"patch_{patch_id}_modis_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MODIS patch with optional HydroRIVERS overlay."
    )
    parser.add_argument("--id", type=int, default=DEFAULT_PATCH_ID)
    parser.add_argument("--patches-folder", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--hydrorivers-gdb-path", type=str, default=None)
    parser.add_argument("--hydrorivers-layer", type=str, default=None)
    parser.add_argument("--min-discharge-cms", type=float, default=0.0)
    parser.add_argument("--min-strahler-order", type=int, default=1)
    parser.add_argument("--discharge-cap-cms", type=float, default=1000.0)
    parser.add_argument("--river-dilate-pixels", type=int, default=1)
    args = parser.parse_args()
    visualize_single_modis(
        patch_id=args.id,
        patches_folder=args.patches_folder,
        metadata_path=args.metadata_path,
        hydrorivers_gdb_path=args.hydrorivers_gdb_path,
        hydrorivers_layer=args.hydrorivers_layer,
        min_discharge_cms=args.min_discharge_cms,
        min_strahler_order=args.min_strahler_order,
        discharge_cap_cms=args.discharge_cap_cms,
        river_dilate_pixels=args.river_dilate_pixels,
    )
