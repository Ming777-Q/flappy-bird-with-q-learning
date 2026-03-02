from __future__ import annotations

from pathlib import Path

# Using pathlib to dynamically resolve absolute paths ensures the project
# is highly portable and runs smoothly on any OS (Windows, macOS, Linux)
# without hardcoding directory structures.

def project_root() -> Path:
    # Resolve the root directory of the project dynamically based on this file's location
    return Path(__file__).resolve().parents[1]

# Define base directories
ROOT_DIR = project_root()
DATA_DIR = ROOT_DIR / "data"

# Define specific file paths for saving/loading RL artifacts
# QVALUES_PATH: The JSON file where the trained Q-table is persisted.
QVALUES_PATH = DATA_DIR / "qvalues.json"
# HITMASKS_PATH: The pickle file containing pre-calculated pixel-perfect collision matrices.
HITMASKS_PATH = DATA_DIR / "hitmasks_data.pkl"

# Define paths for PyGame visualization assets (images and sounds)
ASSETS_DIR = DATA_DIR / "assets"
SPRITES_DIR = ASSETS_DIR / "sprites"
AUDIO_DIR = ASSETS_DIR / "audio"