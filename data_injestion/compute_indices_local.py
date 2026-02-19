import argparse
import json
import os
import re
from glob import glob

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from scipy.ndimage import binary_dilation


# GEE writes MODIS Sinusoidal TIFFs with +datum=WGS84 (ellipsoid), but the
# actual pixel grid uses the MODIS sphere (R=6371007.181m).  All coordinate
# transforms for MODIS data must use this sphere CRS to avoid a ~30-pixel
# vertical shift at mid-latitudes.
MODIS_SINUSOIDAL_SPHERE = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)


def _effective_crs(src):
    """Return the CRS to use for coordinate transforms.

    For MODIS Sinusoidal rasters whose CRS incorrectly specifies the WGS84
    ellipsoid, return the sphere-based CRS.  For everything else (e.g. Landsat
    UTM) return `src.crs` unchanged.
    """
    proj4 = src.crs.to_proj4() if src.crs else ""
    if "+proj=sinu" in proj4:
        return MODIS_SINUSOIDAL_SPHERE
    return src.crs


def _resolve_hydrorivers_layer(gdb_path, layer_name):
    from pyogrio import list_layers

    if layer_name:
        return layer_name

    layers = list_layers(gdb_path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in geodatabase: {gdb_path}")
    return layers[0][0]


def _build_hydrorivers_weighted_channel(
    src,
    gdb_path,
    layer_name,
    where_clause,
    discharge_cap_cms,
    dilate_pixels,
):
    from pyogrio import read_arrow
    from shapely import from_wkb
    from shapely.geometry import mapping

    target_crs = _effective_crs(src)

    bounds_wgs84 = transform_bounds(target_crs, "EPSG:4326", *src.bounds)
    meta, table = read_arrow(
        gdb_path,
        layer=layer_name,
        columns=["DIS_AV_CMS", "ORD_STRA"],
        bbox=bounds_wgs84,
        where=where_clause,
    )

    geometry_col = meta["geometry_name"]
    if geometry_col not in table.column_names or table.num_rows == 0:
        return np.zeros((src.height, src.width), dtype=np.float32)

    cap = max(float(discharge_cap_cms), 1e-6)
    cap_log = np.log1p(cap)

    geom_wkb_list = table[geometry_col].to_pylist()
    dis_list = table["DIS_AV_CMS"].to_pylist()
    ord_list = table["ORD_STRA"].to_pylist()

    shapes = []
    for geom_wkb, dis, ord_stra in zip(geom_wkb_list, dis_list, ord_list):
        if geom_wkb is None:
            continue

        dis = float(dis) if dis is not None else 0.0
        ord_stra = int(ord_stra) if ord_stra is not None else 0

        discharge_weight = np.log1p(max(dis, 0.0)) / cap_log
        discharge_weight = float(np.clip(discharge_weight, 0.0, 1.0))

        if discharge_weight == 0.0 and ord_stra > 0:
            discharge_weight = float(np.clip(ord_stra / 10.0, 0.0, 1.0) * 0.15)

        geom_wgs84 = mapping(from_wkb(geom_wkb))
        geom_local = transform_geom("EPSG:4326", target_crs, geom_wgs84)
        shapes.append((geom_local, discharge_weight))

    if not shapes:
        return np.zeros((src.height, src.width), dtype=np.float32)

    river_weight = np.zeros((src.height, src.width), dtype=np.float32)
    for geom_local, weight in shapes:
        burned = rasterize(
            shapes=[(geom_local, float(weight))],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0.0,
            all_touched=True,
            dtype=np.float32,
        )
        river_weight = np.maximum(river_weight, burned)

    if dilate_pixels > 0:
        river_presence = binary_dilation(
            (river_weight > 0).astype(bool), iterations=dilate_pixels
        )
        river_weight = np.where(river_presence, river_weight, 0.0).astype(np.float32)

    return river_weight.astype(np.float32)


def compute_indices(
    input_tif,
    output_tif,
    include_hydrorivers_mask=False,
    hydrorivers_gdb_path=None,
    hydrorivers_layer=None,
    min_discharge_cms=0.0,
    min_strahler_order=1,
    discharge_cap_cms=1000.0,
    river_dilate_pixels=1,
):
    with rasterio.open(input_tif) as src:
        if src.count < 4:
            raise ValueError(f"{input_tif} has {src.count} band(s); expected at least 4")

        if src.count >= 5:
            # 5-band: Green, Red, NIR, SWIR1, SWIR2 (from patch_add_green)
            green = src.read(1).astype(np.float32)
            red = src.read(2).astype(np.float32)
            nir = src.read(3).astype(np.float32)
            swir1 = src.read(4).astype(np.float32)
            swir2 = src.read(5).astype(np.float32)
        else:
            # 4-band: Red, NIR, SWIR1, SWIR2
            green = None
            red = src.read(1).astype(np.float32)
            nir = src.read(2).astype(np.float32)
            swir1 = src.read(3).astype(np.float32)
            swir2 = src.read(4).astype(np.float32)

        raw_stack = [red, nir, swir1, swir2]
        if green is not None:
            raw_stack.insert(0, green)
        raw_stack = np.stack(raw_stack, axis=0)
        invalid = np.any(~np.isfinite(raw_stack), axis=0)
        nodata = src.nodata
        if nodata is not None:
            invalid |= np.any(np.isclose(raw_stack, nodata), axis=0)
        invalid |= np.all(np.abs(np.nan_to_num(raw_stack, nan=0.0)) <= 1e-8, axis=0)
        patch_cloud_pct = float(invalid.mean() * 100.0)

        # Clean input bands first - replace inf with 0
        red = np.nan_to_num(red, nan=0.0, posinf=0.0, neginf=0.0)
        nir = np.nan_to_num(nir, nan=0.0, posinf=0.0, neginf=0.0)
        swir1 = np.nan_to_num(swir1, nan=0.0, posinf=0.0, neginf=0.0)
        swir2 = np.nan_to_num(swir2, nan=0.0, posinf=0.0, neginf=0.0)
        if green is not None:
            green = np.nan_to_num(green, nan=0.0, posinf=0.0, neginf=0.0)
        
        ndvi = (nir - red) / np.where((nir + red) == 0, np.nan, (nir + red))
        ndmi = (nir - swir1) / np.where((nir + swir1) == 0, np.nan, (nir + swir1))
        water_mask = ((ndvi < 0.1) & (ndmi < 0)).astype(np.float32)

        # MNDWI = (Green - SWIR1) / (Green + SWIR1)
        if green is not None:
            mndwi = (green - swir1) / np.where((green + swir1) == 0, np.nan, (green + swir1))
            mndwi = np.nan_to_num(mndwi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            mndwi = np.zeros_like(ndvi, dtype=np.float32)

        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ndmi = np.nan_to_num(ndmi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        hydrorivers_channel = None
        if include_hydrorivers_mask:
            if not hydrorivers_gdb_path:
                raise ValueError(
                    "HydroRIVERS channel enabled but no geodatabase path was provided."
                )

            where_clause = (
                f"DIS_AV_CMS >= {float(min_discharge_cms)} "
                f"OR ORD_STRA >= {int(min_strahler_order)}"
            )
            hydrorivers_channel = _build_hydrorivers_weighted_channel(
                src=src,
                gdb_path=hydrorivers_gdb_path,
                layer_name=hydrorivers_layer,
                where_clause=where_clause,
                discharge_cap_cms=discharge_cap_cms,
                dilate_pixels=int(river_dilate_pixels),
            )

        profile = src.profile.copy()
        out_count = 9 if include_hydrorivers_mask else 8
        profile.update(dtype=rasterio.float32, count=out_count, compress="deflate")

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(red, 1)
            dst.write(nir, 2)
            dst.write(swir1, 3)
            dst.write(swir2, 4)
            dst.write(water_mask, 5)
            dst.write(ndvi, 6)
            dst.write(ndmi, 7)
            dst.write(mndwi, 8)
            if include_hydrorivers_mask:
                dst.write(hydrorivers_channel, 9)

    return patch_cloud_pct


def _patch_id_from_path(path: str):
    name = os.path.basename(path)
    match = re.search(r"patch_(\d+)\.tif$", name)
    return int(match.group(1)) if match else None


def _update_patch_cloud_metadata(input_folder: str, patch_cloud_pct_by_id: dict):
    metadata_json = os.path.join(input_folder, "metadata.json")
    metadata_csv = os.path.join(input_folder, "metadata.csv")
    if not os.path.exists(metadata_json):
        print(f"metadata.json not found in {input_folder}; skipping metadata cloud update.")
        return

    with open(metadata_json, "r") as f:
        meta = json.load(f)

    updated = 0
    for item in meta:
        pid = int(item.get("patch_id", -1))
        if pid in patch_cloud_pct_by_id:
            item["patch_cloud_coverage_pct"] = float(patch_cloud_pct_by_id[pid])
            updated += 1

    with open(metadata_json, "w") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(metadata_csv):
        try:
            import pandas as pd

            pd.json_normalize(meta).to_csv(metadata_csv, index=False)
        except Exception as exc:
            print(f"Could not update metadata.csv: {exc}")

    print(
        f"Updated metadata with patch_cloud_coverage_pct for {updated} patches: "
        f"{metadata_json}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute water_mask, NDVI, NDMI for local Landsat or MODIS patch tifs. "
            "Use --sensor to select the source; default paths and HydroRIVERS dilation "
            "are adjusted automatically."
        )
    )
    parser.add_argument(
        "--sensor",
        choices=["landsat", "modis"],
        default="landsat",
        help=(
            "Sensor type.  Affects default input/output folders; "
            "HydroRIVERS dilation defaults to 1 pixel for both sensors."
        ),
    )
    parser.add_argument("--input-folder", type=str, default=None)
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=(
            "Folder for output tifs (8 bands by default; 9 bands when "
            "--include-hydrorivers-mask is enabled)."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="patch_*.tif",
        help="Glob pattern inside input folder.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-hydrorivers-mask",
        action="store_true",
        help="Append HydroRIVERS-derived weighted river channel as an extra band.",
    )
    parser.add_argument(
        "--hydrorivers-gdb-path",
        type=str,
        default="data/HydroRIVERS_v10_af.gdb",
        help="Path to HydroRIVERS geodatabase.",
    )
    parser.add_argument(
        "--hydrorivers-layer",
        type=str,
        default=None,
        help="HydroRIVERS layer name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--min-discharge-cms",
        type=float,
        default=0.0,
        help="Include reaches with DIS_AV_CMS >= this threshold.",
    )
    parser.add_argument(
        "--min-strahler-order",
        type=int,
        default=1,
        help="Include reaches with ORD_STRA >= this threshold.",
    )
    parser.add_argument(
        "--discharge-cap-cms",
        type=float,
        default=1000.0,
        help="Cap used to normalize log-scaled DIS_AV_CMS into [0,1].",
    )
    parser.add_argument(
        "--river-dilate-pixels",
        type=int,
        default=None,
        help=(
            "Morphological dilation radius (in pixels) applied to river presence.  "
            "Defaults to 1 for both Landsat and MODIS.  At 500 m/pixel a river "
            "centerline is at most 1–2 pixels wide; dilation=1 widens it to ~3 pixels "
            "(~1.5 km), which is appropriate for the 50 km × 50 km MODIS scene."
        ),
    )
    # MODIS-specific: which dataset sub-folder to process
    parser.add_argument(
        "--modis-dataset",
        choices=["training", "submission", "both"],
        default="training",
        help=(
            "For --sensor=modis: which dataset sub-folder(s) under data/modis_patches "
            "to process.  Ignored when --sensor=landsat or when --input-folder is "
            "specified explicitly."
        ),
    )
    args = parser.parse_args()

    # Fill in sensor-specific defaults for any flag the user left unset.
    if args.sensor == "modis":
        if args.river_dilate_pixels is None:
            args.river_dilate_pixels = 1
        if args.input_folder is None:
            args.input_folder = f"data/modis_patches/{args.modis_dataset}"
        if args.output_folder is None:
            args.output_folder = f"data/modis_patches_with_indices/{args.modis_dataset}"
    else:  # landsat
        if args.river_dilate_pixels is None:
            args.river_dilate_pixels = 1
        if args.input_folder is None:
            args.input_folder = "data/river_patches"
        if args.output_folder is None:
            args.output_folder = "data/river_patches_with_indices"

    # For MODIS "both", recurse into training/ and submission/ sub-folders.
    if args.sensor == "modis" and args.modis_dataset == "both":
        for sub in ("training", "submission"):
            sub_in = os.path.join("data/modis_patches", sub)
            sub_out = os.path.join("data/modis_patches_with_indices", sub)
            if not os.path.isdir(sub_in):
                print(f"Skipping {sub_in} (not found)")
                continue
            _run_folder(
                input_folder=sub_in,
                output_folder=sub_out,
                pattern=args.pattern,
                overwrite=args.overwrite,
                include_hydrorivers_mask=args.include_hydrorivers_mask,
                hydrorivers_gdb_path=args.hydrorivers_gdb_path,
                hydrorivers_layer_hint=args.hydrorivers_layer,
                min_discharge_cms=args.min_discharge_cms,
                min_strahler_order=args.min_strahler_order,
                discharge_cap_cms=args.discharge_cap_cms,
                river_dilate_pixels=args.river_dilate_pixels,
                sensor=args.sensor,
            )
        return

    _run_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        overwrite=args.overwrite,
        include_hydrorivers_mask=args.include_hydrorivers_mask,
        hydrorivers_gdb_path=args.hydrorivers_gdb_path,
        hydrorivers_layer_hint=args.hydrorivers_layer,
        min_discharge_cms=args.min_discharge_cms,
        min_strahler_order=args.min_strahler_order,
        discharge_cap_cms=args.discharge_cap_cms,
        river_dilate_pixels=args.river_dilate_pixels,
        sensor=args.sensor,
    )


def _run_folder(
    input_folder,
    output_folder,
    pattern,
    overwrite,
    include_hydrorivers_mask,
    hydrorivers_gdb_path,
    hydrorivers_layer_hint,
    min_discharge_cms,
    min_strahler_order,
    discharge_cap_cms,
    river_dilate_pixels,
    sensor="landsat",
):
    os.makedirs(output_folder, exist_ok=True)
    tif_paths = sorted(glob(os.path.join(input_folder, pattern)))

    hydrorivers_layer = None
    if include_hydrorivers_mask:
        hydrorivers_layer = _resolve_hydrorivers_layer(
            hydrorivers_gdb_path, hydrorivers_layer_hint
        )
        print(
            "HydroRIVERS weighted channel enabled: "
            f"gdb={hydrorivers_gdb_path}, layer={hydrorivers_layer}, "
            f"where=(DIS_AV_CMS >= {min_discharge_cms} OR ORD_STRA >= {min_strahler_order}), "
            f"discharge_cap_cms={discharge_cap_cms}, "
            f"dilate_pixels={river_dilate_pixels} "
            f"(sensor={sensor})"
        )

    print(f"Sensor: {sensor} | Found {len(tif_paths)} tif file(s) in {input_folder}")
    done = 0
    skipped = 0
    failed = 0
    patch_cloud_pct_by_id = {}

    for i, tif_path in enumerate(tif_paths, start=1):
        out_path = os.path.join(output_folder, os.path.basename(tif_path))
        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            patch_cloud_pct = compute_indices(
                tif_path,
                out_path,
                include_hydrorivers_mask=include_hydrorivers_mask,
                hydrorivers_gdb_path=hydrorivers_gdb_path,
                hydrorivers_layer=hydrorivers_layer,
                min_discharge_cms=min_discharge_cms,
                min_strahler_order=min_strahler_order,
                discharge_cap_cms=discharge_cap_cms,
                river_dilate_pixels=river_dilate_pixels,
            )
            patch_id = _patch_id_from_path(tif_path)
            if patch_id is not None:
                patch_cloud_pct_by_id[patch_id] = float(patch_cloud_pct)
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
    if patch_cloud_pct_by_id:
        _update_patch_cloud_metadata(input_folder, patch_cloud_pct_by_id)


if __name__ == "__main__":
    main()
