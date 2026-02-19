"""Configuration for Water Quality Regression Model"""

from pathlib import Path
import random
import numpy as np
import torch


class Config:
    """Configuration class for water quality regression"""
    
    # Reproducibility
    SEED = 42
    
    # Paths
    ROOT = Path(".")
    CSV_PATH = ROOT / "data" / "water_quality_training_dataset.csv"
    SOILGRIDS_CSV_PATH = ROOT / "data" / "soilgrids_features_full.csv"
    TC_FEATURES_ALL_CSV_PATH = ROOT / "data" / "tc_features_all.csv"
    META_PATH = ROOT / "data" / "river_patches" / "metadata.json"
    IDX_DIR = ROOT / "data" / "river_patches_with_indices"
    VALID_PATCHES_CACHE = ROOT / "data" / "river_patches" / "valid_patches.json"
    LANDCOVERAGE_DIR = ROOT / "data_injestion" / "landcoerage location jsons"
    # Deprecated (kept only for compatibility with existing calls; ignored by loader).
    CLOUD_COVERAGE_CACHE = None
    VALID_INDEXES_CACHE = None
    
    # Image configuration
    INDEX_BANDS = [6, 8, 7, 9]  # [NDVI, MNDWI, NDMI, HydroRIVERS weighted]
    IMG_SIZE = 224
    
    # Feature configuration
    USE_IMAGES = True
    USE_TABULAR = True
    USE_DATETIME_FEATURES = True
    INCLUDE_LANDCOVERAGE_CHANNEL = False
    
    # Target columns
    TARGET_COLS = [
        "Total Alkalinity",
        "Electrical Conductance",
        "Dissolved Reactive Phosphorus",
    ]
    
    # Data filtering
    MAX_DAYS_DIFF = 120
    CLOUD_THRESHOLD_PCT = 20.0
    
    # Cross-validation settings
    N_FOLDS = 5
    TEST_SIZE = 0.2
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 6
    MAX_EPOCHS = 40
    PATIENCE = 8
    LR = 2e-4
    WEIGHT_DECAY = 2e-3
    
    # Model hyperparameters
    TAB_DROPOUT = 0.3
    HEAD_DROPOUT = 0.4
    
    @classmethod
    def set_seed(cls):
        """Set random seeds for reproducibility"""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
    
    @classmethod
    def get_device(cls):
        """Get the device (cuda or cpu)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def print_config(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")