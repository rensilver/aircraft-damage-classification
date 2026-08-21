"""Download and extract the aircraft damage dataset.

Run with ``uv run python scripts/fetch_dataset.py``. If you already have the
dataset extracted elsewhere, pass ``--link-from`` to symlink it instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
    "/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "aircraft_damage_dataset_v1"
EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_CLASSES = ("crack", "dent")


def verify(dataset_dir: Path) -> None:
    """Check the extracted layout and print per-split counts.

    Args:
        dataset_dir: Root of the extracted dataset.

    Raises:
        SystemExit: If an expected split or class folder is missing.
    """
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            folder = dataset_dir / split / class_name
            if not folder.is_dir():
                sys.exit(f"Missing expected folder: {folder}")
            count = len(list(folder.glob("*.jpg")))
            print(f"  {split}/{class_name}: {count} images")


def link_from(source: Path) -> None:
    """Symlink an existing extracted dataset into ``data/``.

    Args:
        source: Path to an existing ``aircraft_damage_dataset_v1`` directory.

    Raises:
        SystemExit: If the source does not exist.
    """
    if not source.is_dir():
        sys.exit(f"Source directory does not exist: {source}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists() or DATASET_DIR.is_symlink():
        sys.exit(f"{DATASET_DIR} already exists; remove it first.")
    DATASET_DIR.symlink_to(source.resolve(), target_is_directory=True)
    print(f"Linked {DATASET_DIR} -> {source.resolve()}")


def download() -> None:
    """Download the tarball and extract it into ``data/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "aircraft_damage_dataset_v1.tar"

    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, tar_path)  # noqa: S310

    if DATASET_DIR.exists():
        print(f"Removing existing {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    print(f"Extracting into {DATA_DIR}")
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(DATA_DIR, filter="data")
    tar_path.unlink()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link-from",
        type=Path,
        default=None,
        help="Symlink an already-extracted dataset instead of downloading.",
    )
    args = parser.parse_args()

    if args.link_from is not None:
        link_from(args.link_from)
    elif DATASET_DIR.exists():
        print(f"{DATASET_DIR} already exists; skipping download.")
    else:
        download()

    print("Dataset layout:")
    verify(DATASET_DIR)


if __name__ == "__main__":
    main()
