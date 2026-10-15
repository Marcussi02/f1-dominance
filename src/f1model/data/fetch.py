"""Download the F1DB open dataset (1950 to now) and cache its tables as Parquet.

F1DB is published as a release asset on GitHub: https://github.com/f1db/f1db
The zip is read in place, never extracted, and each table is converted to
Parquet the first time it is asked for.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

F1DB_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("F1MODEL_DATA", PROJECT_ROOT / "data"))
RAW_ZIP = DATA_DIR / "raw" / "f1db-csv.zip"
CACHE_DIR = DATA_DIR / "cache"

TABLES = {
    "races": "f1db-races.csv",
    "results": "f1db-races-race-results.csv",
    "qualifying": "f1db-races-qualifying-results.csv",
    "circuits": "f1db-circuits.csv",
    "drivers": "f1db-drivers.csv",
    "constructors": "f1db-constructors.csv",
    "entrants": "f1db-seasons-entrants-drivers.csv",
}


def download(force: bool = False) -> Path:
    """Fetch the F1DB zip unless a copy is already on disk."""
    if RAW_ZIP.exists() and not force:
        return RAW_ZIP
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(F1DB_URL, timeout=120)
    resp.raise_for_status()
    tmp = RAW_ZIP.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(RAW_ZIP)
    # A new download makes every cached table stale.
    for old in CACHE_DIR.glob("*.parquet"):
        old.unlink()
    return RAW_ZIP


def load_table(name: str) -> pd.DataFrame:
    """Return one F1DB table, building its Parquet cache on first use."""
    cached = CACHE_DIR / f"{name}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    with zipfile.ZipFile(download()) as zf:
        with zf.open(TABLES[name]) as fh:
            df = pd.read_csv(fh, low_memory=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    return df
