"""
Download GazeBase dataset from Figshare.

Usage:
    python scripts/download_data.py --output_dir data/gazebase
    python scripts/download_data.py --source rounds --rounds Round_1 Round_2

GazeBase repository: https://figshare.com/articles/dataset/GazeBase_data_repository/12912257
"""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


# Current Figshare file selected from the article page (?file=27039812).
GAZEBASE_ARCHIVE_URL = "https://figshare.com/ndownloader/files/27039812"
GAZEBASE_ARCHIVE_NAME = "gazebase_27039812.zip"

# GazeBase Figshare file URLs (Round 1-9)
GAZEBASE_URLS = {
    "Round_1": "https://figshare.com/ndownloader/files/26798386",
    "Round_2": "https://figshare.com/ndownloader/files/26798395",
    "Round_3": "https://figshare.com/ndownloader/files/26798398",
    "Round_4": "https://figshare.com/ndownloader/files/26798401",
    "Round_5": "https://figshare.com/ndownloader/files/26798404",
    "Round_6": "https://figshare.com/ndownloader/files/26798407",
    "Round_7": "https://figshare.com/ndownloader/files/26798413",
    "Round_8": "https://figshare.com/ndownloader/files/26798416",
    "Round_9": "https://figshare.com/ndownloader/files/26798419",
}


class DownloadProgress:
    """Progress bar for downloads."""

    def __init__(self, desc: str):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=self.desc)
        downloaded = block_num * block_size
        self.pbar.update(block_size)
        if downloaded >= total_size and self.pbar:
            self.pbar.close()


def download_gazebase(output_dir: str, rounds: list[str] = None):
    """
    Download GazeBase dataset.

    Args:
        output_dir: Directory to save data.
        rounds: Which rounds to download (default: all).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if rounds is None:
        rounds = list(GAZEBASE_URLS.keys())

    print(f"Downloading GazeBase to: {output_path}")
    print(f"Rounds: {rounds}")
    print()

    for round_name in rounds:
        round_dir = output_path / round_name
        if round_dir.exists() and any(round_dir.glob("*.csv")):
            print(f"  {round_name}: already exists, skipping")
            continue

        url = GAZEBASE_URLS.get(round_name)
        if not url:
            print(f"  {round_name}: URL not found, skipping")
            continue

        zip_path = output_path / f"{round_name}.zip"

        # Download
        print(f"  Downloading {round_name}...")
        try:
            urlretrieve(url, zip_path, reporthook=DownloadProgress(round_name))
        except Exception as e:
            print(f"  Error downloading {round_name}: {e}")
            print(f"  Try manually downloading from: {url}")
            continue

        # Extract
        print(f"  Extracting {round_name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_path)
            zip_path.unlink()  # Remove zip after extraction
            print(f"  {round_name}: done")
        except Exception as e:
            print(f"  Error extracting {round_name}: {e}")

    # Summary
    print("\nDataset summary:")
    total_files = 0
    for round_dir in sorted(output_path.glob("Round_*")):
        n_files = len(list(round_dir.glob("*.csv")))
        total_files += n_files
        print(f"  {round_dir.name}: {n_files} files")
    print(f"  Total: {total_files} files")


def extract_zip(zip_path: Path, output_path: Path, keep_zip: bool = False):
    """Extract a downloaded zip and optionally remove it afterward."""
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty or missing: {zip_path}")

    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_path)

    if not keep_zip:
        zip_path.unlink()


def print_dataset_summary(output_path: Path):
    """Print a short CSV summary after download/extraction."""
    print("\nDataset summary:")
    total_files = 0
    round_dirs = sorted(output_path.rglob("Round_*"))

    if round_dirs:
        for round_dir in round_dirs:
            if not round_dir.is_dir():
                continue
            n_files = len(list(round_dir.glob("*.csv")))
            total_files += n_files
            print(f"  {round_dir.relative_to(output_path)}: {n_files} files")
    else:
        total_files = len(list(output_path.rglob("*.csv")))

    print(f"  Total CSV files: {total_files}")


def download_gazebase_archive(output_dir: str, keep_zip: bool = False):
    """Download the current Figshare archive file selected by file=27039812."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / GAZEBASE_ARCHIVE_NAME

    print(f"Downloading current GazeBase archive to: {output_path}")
    print(f"URL: {GAZEBASE_ARCHIVE_URL}")
    print()

    if any(output_path.rglob("*.csv")):
        print("CSV files already exist under output_dir; skipping download.")
        print_dataset_summary(output_path)
        return

    try:
        urlretrieve(
            GAZEBASE_ARCHIVE_URL,
            zip_path,
            reporthook=DownloadProgress(GAZEBASE_ARCHIVE_NAME),
        )
        extract_zip(zip_path, output_path, keep_zip=keep_zip)
    except Exception as e:
        print(f"Error downloading/extracting archive: {e}")
        print(f"Try manually downloading from: {GAZEBASE_ARCHIVE_URL}")
        raise

    print_dataset_summary(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download GazeBase dataset")
    parser.add_argument("--output_dir", type=str, default="data/gazebase")
    parser.add_argument(
        "--source",
        choices=["archive", "rounds"],
        default="archive",
        help="archive uses current Figshare file=27039812; rounds uses old Round_1-9 URLs",
    )
    parser.add_argument(
        "--rounds", type=str, nargs="+", default=None,
        help="Specific rounds to download (e.g., Round_1 Round_2)"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )
    args = parser.parse_args()

    if args.source == "archive":
        download_gazebase_archive(args.output_dir, keep_zip=args.keep_zip)
    else:
        download_gazebase(args.output_dir, args.rounds)


if __name__ == "__main__":
    main()
