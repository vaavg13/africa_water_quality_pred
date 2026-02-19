"""Data loading and dataset classes for water quality regression"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config


def load_and_prepare_dataframe(
    csv_path: Path,
    meta_path: Path,
    idx_dir: Path,
    target_cols: list,
    index_bands: list,
    soilgrids_csv_path: Path = Config.SOILGRIDS_CSV_PATH,
    tc_features_csv_path: Path = Config.TC_FEATURES_ALL_CSV_PATH,
    use_datetime_features: bool = False,
    max_days_diff: int = Config.MAX_DAYS_DIFF,
    valid_patches_cache: Path = Config.VALID_PATCHES_CACHE,
    cloud_threshold_pct: float = Config.CLOUD_THRESHOLD_PCT,
    cloud_coverage_cache: Path = None,
    valid_indexes_cache: Path = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Load and prepare the dataframe with water quality data.
    
    Args:
        csv_path: Path to CSV file with labels
        meta_path: Path to metadata JSON file
        idx_dir: Directory containing index TIF files
        target_cols: List of target column names
        index_bands: List of band indices to use
        tc_features_csv_path: Path to unified TC tabular features CSV
        use_datetime_features: Whether to include datetime features
        max_days_diff: Maximum days difference for filtering
        valid_patches_cache: Path to cache file for valid patches
        cloud_threshold_pct: Optional maximum allowed cloud coverage percentage
        cloud_coverage_cache: Deprecated/ignored. Cloud filtering is recomputed each run.
        valid_indexes_cache: Deprecated/ignored. Date filtering is from metadata each run.
        
    Returns:
        Tuple of (dataframe, list of tabular column names)
    """
    labels = pd.read_csv(csv_path).reset_index().rename(columns={"index": "dataset_index"})

    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta_df = pd.json_normalize(meta, sep="_")

    needed_meta = [
        "dataset_index", "patch_id", "file", "lat", "lon", "target_date",
        "target_date_iso", "actual_date", "days_difference",
        "download_status", "cloud_cover",
    ]
    missing_cols = [c for c in needed_meta if c not in meta_df.columns]
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {missing_cols}")

    meta_df = meta_df[meta_df["download_status"].isin(["downloaded", "exists"])].copy()
    meta_df["days_difference"] = pd.to_numeric(meta_df["days_difference"], errors="coerce")

    meta_df["idx_tif"] = meta_df["patch_id"].apply(lambda x: idx_dir / f"patch_{int(x)}.tif")
    meta_df = meta_df[meta_df["idx_tif"].apply(lambda p: Path(p).exists())].copy()

    if cloud_coverage_cache is not None:
        print("cloud_coverage_cache argument is ignored; cloud coverage is recomputed each run.")
    if valid_indexes_cache is not None:
        print("valid_indexes_cache argument is ignored; date filtering uses metadata each run.")

    # Filter out patches with completely empty index bands (cached)
    valid_patch_ids = set()
    if valid_patches_cache and valid_patches_cache.exists():
        print(f"Loading valid patches cache from {valid_patches_cache}")
        with open(valid_patches_cache, "r") as f:
            valid_patch_ids = set(json.load(f))
    else:
        print("Building valid patches cache (this may take a while)...")
        print("Note: This cache is independent of cloud/date filtering and will be reused.")

        def has_valid_indices(tif_path):
            """Check if at least one selected band has non-zero data."""
            try:
                with rasterio.open(tif_path) as src:
                    for band_idx in index_bands:
                        if band_idx <= src.count:
                            band = src.read(band_idx)
                            if np.count_nonzero(band) > 0:
                                return True
                return False
            except Exception:
                return False

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Checking patches"):
            if has_valid_indices(row["idx_tif"]):
                valid_patch_ids.add(int(row["patch_id"]))

        if valid_patches_cache:
            valid_patches_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(valid_patches_cache, "w") as f:
                json.dump(list(valid_patch_ids), f)
            print(f"Saved valid patches cache to {valid_patches_cache}")

    initial_count = len(meta_df)
    meta_df = meta_df[meta_df["patch_id"].isin(valid_patch_ids)].copy()
    filtered_count = initial_count - len(meta_df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} patches with completely empty index bands")

    # Optional cloud filtering.
    if cloud_threshold_pct is not None:
        cloud_threshold_pct = float(cloud_threshold_pct)
        cloud_source = None
        if "patch_cloud_coverage_pct" in meta_df.columns:
            meta_df["cloud_pct_computed"] = pd.to_numeric(
                meta_df["patch_cloud_coverage_pct"], errors="coerce"
            )
            cloud_source = "patch_cloud_coverage_pct"
        else:
            # Fallback to scene-level CLOUD_COVER from metadata.
            meta_df["cloud_pct_computed"] = pd.to_numeric(meta_df["cloud_cover"], errors="coerce")
            cloud_source = "cloud_cover (scene-level fallback)"

        print(
            f"Applying cloud threshold from metadata field '{cloud_source}' "
            f"(threshold={cloud_threshold_pct:.1f}%)..."
        )

        before_cloud = len(meta_df)
        meta_df = meta_df[meta_df["cloud_pct_computed"] <= cloud_threshold_pct].copy()
        cloud_filtered = before_cloud - len(meta_df)
        if cloud_filtered > 0:
            print(
                f"Filtered out {cloud_filtered} patches with computed cloud coverage > "
                f"{cloud_threshold_pct:.1f}%"
            )

    # Apply time filtering directly from metadata (no cache by design)
    if max_days_diff is not None:
        if len(meta_df) > 0:
            min_dd = float(meta_df["days_difference"].min())
            max_dd = float(meta_df["days_difference"].max())
            print(f"days_difference range before time filter: min={min_dd:.2f}, max={max_dd:.2f}")
        initial_time_count = len(meta_df)
        meta_df = meta_df[meta_df["days_difference"] <= max_days_diff].copy()
        time_filtered = initial_time_count - len(meta_df)
        if time_filtered > 0:
            print(f"Filtered out {time_filtered} patches with days_difference > {max_days_diff}")
        else:
            print(
                f"No patches exceeded max_days_diff={max_days_diff}; "
                "time filter kept all currently eligible rows."
            )

    df = meta_df.merge(labels, on="dataset_index", how="inner", suffixes=("_meta", ""))

    if "Sample Date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["Sample Date"], format="%d-%m-%Y", errors="coerce")
    else:
        df["sample_date"] = pd.to_datetime(df["target_date_iso"], errors="coerce")

    # Temporal encoding: day-of-year seasonality (sin/cos)
    df["day_of_year"] = df["sample_date"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    for t in target_cols:
        if t not in df.columns:
            raise ValueError(f"Target column missing: {t}")

    tab_cols = []

    # Merge external TC weather/hydrology features by location + sample date.
    df["lat_join"] = pd.to_numeric(df["lat"], errors="coerce").round(6)
    df["lon_join"] = pd.to_numeric(df["lon"], errors="coerce").round(6)
    df["date_join"] = pd.to_datetime(df["sample_date"], errors="coerce").dt.normalize()

    if tc_features_csv_path is not None and Path(tc_features_csv_path).exists():
        tc_df = pd.read_csv(tc_features_csv_path).copy()
        required_tc_keys = {"Latitude", "Longitude", "Sample Date"}
        if not required_tc_keys.issubset(set(tc_df.columns)):
            raise ValueError(
                f"Missing required columns in unified TC features CSV: "
                f"{sorted(required_tc_keys - set(tc_df.columns))}"
            )

        tc_df["lat_join"] = pd.to_numeric(tc_df["Latitude"], errors="coerce").round(6)
        tc_df["lon_join"] = pd.to_numeric(tc_df["Longitude"], errors="coerce").round(6)
        tc_df["date_join"] = pd.to_datetime(tc_df["Sample Date"], errors="coerce").dt.normalize()

        tc_join_keys = {"lat_join", "lon_join", "date_join"}
        tc_feature_cols = [c for c in tc_df.columns if c not in required_tc_keys.union(tc_join_keys)]
        for col in tc_feature_cols:
            tc_df[col] = pd.to_numeric(tc_df[col], errors="coerce")

        tc_df = tc_df.drop_duplicates(subset=["lat_join", "lon_join", "date_join"], keep="first")
        df = df.merge(
            tc_df[["lat_join", "lon_join", "date_join"] + tc_feature_cols],
            on=["lat_join", "lon_join", "date_join"],
            how="left",
        )
        tab_cols.extend(tc_feature_cols)
    elif tc_features_csv_path is not None:
        print(f"Unified TC features CSV not found at {tc_features_csv_path}; continuing without TC features.")

    df = df.drop(columns=["lat_join", "lon_join", "date_join"])

    # Add a compact, lower-collinearity subset of SoilGrids signals.
    # Keep only properties with strong hydrochemistry relevance and aggregate
    # shallow depths (0-30cm) to avoid injecting highly redundant depth stacks.
    if soilgrids_csv_path is not None and Path(soilgrids_csv_path).exists():
        soil_df = pd.read_csv(soilgrids_csv_path).copy()

        required_soil_cols = {"lat", "lon"}
        if not required_soil_cols.issubset(set(soil_df.columns)):
            raise ValueError(
                f"Missing required SoilGrids columns: {sorted(required_soil_cols - set(soil_df.columns))}"
            )

        soil_df["lat"] = pd.to_numeric(soil_df["lat"], errors="coerce")
        soil_df["lon"] = pd.to_numeric(soil_df["lon"], errors="coerce")

        useful_soil_prefixes = [
            "soil_bdod",   # bulk density proxy for compaction/infiltration
            "soil_cec",    # cation exchange capacity for nutrient buffering
            "soil_clay",   # texture control on transport/retention
            "soil_phh2o",  # baseline soil acidity/alkalinity
            "soil_sand",   # complementary texture signal to clay
            "soil_soc",    # organic carbon proxy for nutrient dynamics
        ]
        topsoil_depths = ["0_5cm", "5_15cm", "15_30cm"]

        engineered_soil_cols = []
        for prefix in useful_soil_prefixes:
            depth_cols = [f"{prefix}_{depth}" for depth in topsoil_depths if f"{prefix}_{depth}" in soil_df.columns]
            if not depth_cols:
                continue
            for depth_col in depth_cols:
                soil_df[depth_col] = pd.to_numeric(soil_df[depth_col], errors="coerce")
            out_col = f"{prefix}_top30cm_mean"
            soil_df[out_col] = soil_df[depth_cols].mean(axis=1)
            engineered_soil_cols.append(out_col)

        if engineered_soil_cols:
            # Coordinate matching can differ in tiny floating-point decimals.
            df["lat_join"] = df["lat"].round(5)
            df["lon_join"] = df["lon"].round(5)
            soil_df["lat_join"] = soil_df["lat"].round(5)
            soil_df["lon_join"] = soil_df["lon"].round(5)

            soil_merge_cols = ["lat_join", "lon_join"] + engineered_soil_cols
            df = df.merge(soil_df[soil_merge_cols], on=["lat_join", "lon_join"], how="left")
            df = df.drop(columns=["lat_join", "lon_join"])
            tab_cols.extend(engineered_soil_cols)
    elif soilgrids_csv_path is not None:
        print(f"SoilGrids CSV not found at {soilgrids_csv_path}; continuing without soil features.")

    datetime_cols = ["day_sin", "day_cos"]
    if use_datetime_features:
        tab_cols.extend(datetime_cols)
    tab_cols = list(dict.fromkeys(tab_cols))

    for col in tab_cols + target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=target_cols).copy()
    for col in tab_cols:
        df[col] = df[col].fillna(df[col].median())

    df["group_id"] = df["lat"].round(4).astype(str) + "_" + df["lon"].round(4).astype(str)

    return df, tab_cols


def load_submission_for_inference(
    submission_csv_path: Path,
    meta_path: Path,
    idx_dir: Path,
    index_bands: list,
    soilgrids_csv_path: Path = Config.SOILGRIDS_CSV_PATH,
    tc_features_csv_path: Path = Config.TC_FEATURES_ALL_CSV_PATH,
    use_datetime_features: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Load submission template and match to metadata for inference.
    No cloud/valid_patches/max_days filtering - uses whatever patches exist.
    """
    sub_df = pd.read_csv(submission_csv_path).copy()
    sub_df["Latitude"] = pd.to_numeric(sub_df["Latitude"], errors="coerce")
    sub_df["Longitude"] = pd.to_numeric(sub_df["Longitude"], errors="coerce")
    sub_df["sample_date"] = pd.to_datetime(sub_df["Sample Date"], format="%d-%m-%Y", errors="coerce")
    sub_df = sub_df.dropna(subset=["Latitude", "Longitude", "sample_date"])

    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta_df = pd.json_normalize(meta, sep="_")

    needed_meta = ["patch_id", "file", "lat", "lon", "target_date", "target_date_iso", "download_status"]
    missing_cols = [c for c in needed_meta if c not in meta_df.columns]
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {missing_cols}")

    meta_df = meta_df[meta_df["download_status"].isin(["downloaded", "exists"])].copy()
    meta_df["idx_tif"] = meta_df["patch_id"].apply(lambda x: idx_dir / f"patch_{int(x)}.tif")
    meta_df = meta_df[meta_df["idx_tif"].apply(lambda p: Path(p).exists())].copy()

    sub_df["lat_join"] = sub_df["Latitude"].round(5)
    sub_df["lon_join"] = sub_df["Longitude"].round(5)
    sub_df["date_join"] = sub_df["sample_date"].dt.normalize()

    meta_df["lat_join"] = pd.to_numeric(meta_df["lat"], errors="coerce").round(5)
    meta_df["lon_join"] = pd.to_numeric(meta_df["lon"], errors="coerce").round(5)
    meta_df["date_join"] = pd.to_datetime(meta_df["target_date_iso"], errors="coerce").dt.normalize()

    df = sub_df.merge(
        meta_df[["lat_join", "lon_join", "date_join", "patch_id", "file", "lat", "lon", "idx_tif", "target_date", "target_date_iso"]],
        on=["lat_join", "lon_join", "date_join"],
        how="inner",
    )
    df = df.drop(columns=["lat_join", "lon_join", "date_join"])

    df["day_of_year"] = df["sample_date"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    tab_cols = []

    if tc_features_csv_path is not None and Path(tc_features_csv_path).exists():
        tc_df = pd.read_csv(tc_features_csv_path).copy()
        required_tc_keys = {"Latitude", "Longitude", "Sample Date"}
        if required_tc_keys.issubset(set(tc_df.columns)):
            tc_df["lat_join"] = pd.to_numeric(tc_df["Latitude"], errors="coerce").round(6)
            tc_df["lon_join"] = pd.to_numeric(tc_df["Longitude"], errors="coerce").round(6)
            tc_df["date_join"] = pd.to_datetime(tc_df["Sample Date"], errors="coerce").dt.normalize()
            tc_feature_cols = [c for c in tc_df.columns if c not in required_tc_keys.union({"lat_join", "lon_join", "date_join"})]
            for col in tc_feature_cols:
                tc_df[col] = pd.to_numeric(tc_df[col], errors="coerce")
            tc_df = tc_df.drop_duplicates(subset=["lat_join", "lon_join", "date_join"], keep="first")
            df["lat_join"] = df["lat"].round(6)
            df["lon_join"] = df["lon"].round(6)
            df["date_join"] = df["sample_date"].dt.normalize()
            df = df.merge(
                tc_df[["lat_join", "lon_join", "date_join"] + tc_feature_cols],
                on=["lat_join", "lon_join", "date_join"],
                how="left",
            )
            df = df.drop(columns=["lat_join", "lon_join", "date_join"])
            tab_cols.extend(tc_feature_cols)

    if soilgrids_csv_path is not None and Path(soilgrids_csv_path).exists():
        soil_df = pd.read_csv(soilgrids_csv_path).copy()
        if {"lat", "lon"}.issubset(set(soil_df.columns)):
            soil_df["lat"] = pd.to_numeric(soil_df["lat"], errors="coerce")
            soil_df["lon"] = pd.to_numeric(soil_df["lon"], errors="coerce")
            useful_soil_prefixes = [
                "soil_bdod", "soil_cec", "soil_clay", "soil_phh2o", "soil_sand", "soil_soc",
            ]
            topsoil_depths = ["0_5cm", "5_15cm", "15_30cm"]
            engineered_soil_cols = []
            for prefix in useful_soil_prefixes:
                depth_cols = [f"{prefix}_{d}" for d in topsoil_depths if f"{prefix}_{d}" in soil_df.columns]
                if not depth_cols:
                    continue
                for dc in depth_cols:
                    soil_df[dc] = pd.to_numeric(soil_df[dc], errors="coerce")
                out_col = f"{prefix}_top30cm_mean"
                soil_df[out_col] = soil_df[depth_cols].mean(axis=1)
                engineered_soil_cols.append(out_col)
            if engineered_soil_cols:
                df["lat_join"] = df["lat"].round(5)
                df["lon_join"] = df["lon"].round(5)
                soil_df["lat_join"] = soil_df["lat"].round(5)
                soil_df["lon_join"] = soil_df["lon"].round(5)
                df = df.merge(
                    soil_df[["lat_join", "lon_join"] + engineered_soil_cols],
                    on=["lat_join", "lon_join"],
                    how="left",
                )
                df = df.drop(columns=["lat_join", "lon_join"])
                tab_cols.extend(engineered_soil_cols)

    if use_datetime_features:
        tab_cols.extend(["day_sin", "day_cos"])
    tab_cols = list(dict.fromkeys(tab_cols))

    for col in tab_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["group_id"] = df["lat"].round(4).astype(str) + "_" + df["lon"].round(4).astype(str)
    return df, tab_cols


class WaterQualityDataset(Dataset):
    """PyTorch Dataset for water quality regression"""
    
    def __init__(
        self,
        frame: pd.DataFrame,
        tab_scaled: np.ndarray,
        y_scaled: np.ndarray,
        index_bands: list,
        img_size: int = 224,
        augment: bool = False,
        use_images: bool = True,
        use_tabular: bool = True,
        landcoverage_dir: Path = Config.LANDCOVERAGE_DIR,
        include_landcoverage_channel: bool = Config.INCLUDE_LANDCOVERAGE_CHANNEL,
    ):
        """
        Args:
            frame: DataFrame with sample metadata
            tab_scaled: Scaled tabular features
            y_scaled: Scaled target values
            index_bands: List of band indices to use
            img_size: Size to resize images to
            augment: Whether to apply data augmentation
            use_images: Whether to use image data
            use_tabular: Whether to use tabular data
        """
        self.frame = frame.reset_index(drop=True)
        self.tab = tab_scaled.astype(np.float32)
        self.y = y_scaled.astype(np.float32)
        self.index_bands = index_bands
        self.img_size = img_size
        self.augment = augment
        self.use_images = use_images
        self.use_tabular = use_tabular
        self.include_landcoverage_channel = include_landcoverage_channel
        self.landcoverage_dir = Path(landcoverage_dir) if landcoverage_dir is not None else None
        self._landcoverage_cache: Dict[int, Optional[np.ndarray]] = {}

    def __len__(self):
        return len(self.frame)

    def _load_landcoverage_grid(self, patch_id: int) -> Optional[np.ndarray]:
        """Load cached land coverage grid (334x334) for a patch id."""
        if self.landcoverage_dir is None:
            return None
        if patch_id in self._landcoverage_cache:
            return self._landcoverage_cache[patch_id]

        json_path = self.landcoverage_dir / f"location_{int(patch_id)}.json"
        if not json_path.exists():
            self._landcoverage_cache[patch_id] = None
            return None

        try:
            with open(json_path, "r") as f:
                payload = json.load(f)
            landcover = payload.get("layers", {}).get("landcover", None)
            if landcover is None:
                self._landcoverage_cache[patch_id] = None
                return None

            grid = np.array(landcover, dtype=np.float32)
            grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
            self._landcoverage_cache[patch_id] = grid
            return grid
        except Exception:
            self._landcoverage_cache[patch_id] = None
            return None

    @staticmethod
    def _center_crop_or_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Center crop or zero-pad 2D array to the requested shape."""
        h, w = arr.shape
        if h >= target_h:
            top = (h - target_h) // 2
            arr = arr[top : top + target_h, :]
        else:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0.0)

        h, w = arr.shape
        if w >= target_w:
            left = (w - target_w) // 2
            arr = arr[:, left : left + target_w]
        else:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)

        return arr.astype(np.float32)

    def _extract_landcoverage_channel(
        self, patch_id: int, target_h: int, target_w: int
    ) -> np.ndarray:
        """
        Build a POI-centered land coverage channel aligned to source patch dimensions.
        """
        grid = self._load_landcoverage_grid(patch_id)
        if grid is None:
            return np.zeros((target_h, target_w), dtype=np.float32)

        centered = self._center_crop_or_pad(grid, target_h, target_w)
        # Keep a compact single nominal channel in [0, 1].
        centered = np.clip(centered / 100.0, 0.0, 1.0)
        return centered.astype(np.float32)

    def _read_indices(self, tif_path: Path, patch_id: int):
        """Read and preprocess index bands from TIF file"""
        with rasterio.open(tif_path) as src:
            nband = src.count
            src_h, src_w = src.height, src.width
            channels = []
            for b in self.index_bands:
                if b <= nband:
                    ch = src.read(b).astype(np.float32)
                else:
                    ch = np.zeros((src.height, src.width), dtype=np.float32)
                channels.append(ch)

            if self.include_landcoverage_channel:
                channels.append(self._extract_landcoverage_channel(patch_id, src_h, src_w))

            arr = np.stack(channels, axis=0)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)

        x = torch.from_numpy(arr)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1).item() < 0.5:
                k = int(torch.randint(0, 4, (1,)).item())
                x = torch.rot90(x, k=k, dims=[1, 2])

        return x

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        
        if self.use_images:
            x_img = self._read_indices(Path(row["idx_tif"]), int(row["patch_id"]))
        else:
            num_channels = len(self.index_bands) + int(self.include_landcoverage_channel)
            x_img = torch.zeros(
                num_channels,
                self.img_size,
                self.img_size,
                dtype=torch.float32
            )
        
        if self.use_tabular:
            x_tab = torch.from_numpy(self.tab[idx])
        else:
            x_tab = torch.zeros(self.tab.shape[1], dtype=torch.float32)
        
        y = torch.from_numpy(self.y[idx])
        return x_img, x_tab, y
