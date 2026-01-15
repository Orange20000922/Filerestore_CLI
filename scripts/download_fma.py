#!/usr/bin/env python3
"""
Free Music Archive (FMA) Dataset Downloader

Downloads MP3 audio files from the FMA dataset for ML training.
FMA contains 106,574 tracks from 16,341 artists and 14,854 albums.

Dataset subsets:
- fma_small:  8,000 tracks (30s clips), ~7.2 GB
- fma_medium: 25,000 tracks (30s clips), ~22 GB
- fma_large:  106,574 tracks (30s clips), ~93 GB
- fma_full:   106,574 tracks (full length), ~879 GB

Usage:
    # Download small subset
    python download_fma.py download --subset small --dest ./data/fma

    # Download to OneDrive
    python download_fma.py download --subset small --dest onedrive://FMA/

    # List available subsets
    python download_fma.py list
"""

import argparse
import os
import sys
import zipfile
import tempfile
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import logging

sys.path.insert(0, str(Path(__file__).parent))
from storage import get_storage

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

FMA_BASE_URL = "https://os.unil.cloud.switch.ch/fma"

FMA_SUBSETS = {
    "small": {
        "url": f"{FMA_BASE_URL}/fma_small.zip",
        "size_gb": 7.2,
        "tracks": 8000,
        "description": "8,000 tracks of 30s, 8 balanced genres"
    },
    "medium": {
        "url": f"{FMA_BASE_URL}/fma_medium.zip",
        "size_gb": 22,
        "tracks": 25000,
        "description": "25,000 tracks of 30s, 16 unbalanced genres"
    },
    "large": {
        "url": f"{FMA_BASE_URL}/fma_large.zip",
        "size_gb": 93,
        "tracks": 106574,
        "description": "106,574 tracks of 30s, 161 unbalanced genres"
    },
    "full": {
        "url": f"{FMA_BASE_URL}/fma_full.zip",
        "size_gb": 879,
        "tracks": 106574,
        "description": "106,574 untrimmed tracks, 161 genres"
    },
    "metadata": {
        "url": f"{FMA_BASE_URL}/fma_metadata.zip",
        "size_gb": 0.3,
        "tracks": 0,
        "description": "Track metadata, tags, and features"
    }
}


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Download Functions
# =============================================================================

def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 1024 * 1024
) -> bool:
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.with_suffix('.tmp')

        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        temp_path.rename(dest_path)
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_mp3_files(
    zip_path: Path,
    output_dir: Path,
    max_files: int = 0
) -> int:
    """Extract MP3 files from FMA zip archive"""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    print(f"Extracting MP3 files from {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        mp3_files = [f for f in zf.namelist() if f.endswith('.mp3')]
        total = len(mp3_files)

        if max_files > 0:
            mp3_files = mp3_files[:max_files]

        for member in tqdm(mp3_files, desc="Extracting"):
            try:
                # Extract to flat directory structure
                filename = Path(member).name
                target_path = output_dir / filename

                with zf.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

                extracted += 1
            except Exception as e:
                logger.warning(f"Failed to extract {member}: {e}")

    return extracted


def download_fma(
    subset: str,
    dest: str,
    max_files: int = 0,
    keep_zip: bool = False,
    skip_existing: bool = True
):
    """
    Download FMA dataset subset

    Args:
        subset: Dataset subset (small, medium, large, full)
        dest: Destination path (local or onedrive://path)
        max_files: Maximum MP3 files to extract (0 = all)
        keep_zip: Keep zip file after extraction
        skip_existing: Skip if destination has files
    """
    if subset not in FMA_SUBSETS:
        print(f"Error: Unknown subset '{subset}'")
        print(f"Available: {list(FMA_SUBSETS.keys())}")
        return

    info = FMA_SUBSETS[subset]
    print(f"\nFMA {subset} subset:")
    print(f"  Tracks: {info['tracks']:,}")
    print(f"  Size: {info['size_gb']} GB")
    print(f"  Description: {info['description']}")

    # Determine storage backend
    if dest.startswith("onedrive://"):
        upload_to_onedrive = True
        onedrive_path = dest[11:]
        local_dest = Path(tempfile.gettempdir()) / "fma_temp"
        storage = get_storage("onedrive")
    else:
        upload_to_onedrive = False
        local_dest = Path(dest)
        storage = None

    local_dest.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_filename = f"fma_{subset}.zip"
    zip_path = local_dest / zip_filename

    if zip_path.exists() and skip_existing:
        print(f"\nZip file already exists: {zip_path}")
    else:
        print(f"\nDownloading {zip_filename}...")
        print(f"This may take a while ({info['size_gb']} GB)")

        if not download_file(info['url'], zip_path):
            print("Download failed!")
            return

    # Extract MP3 files
    mp3_dir = local_dest / f"fma_{subset}_mp3"
    extracted = extract_mp3_files(zip_path, mp3_dir, max_files)
    print(f"\nExtracted {extracted} MP3 files to {mp3_dir}")

    # Upload to OneDrive if needed
    if upload_to_onedrive and storage:
        print(f"\nUploading to OneDrive:{onedrive_path}...")

        mp3_files = list(mp3_dir.glob("*.mp3"))
        uploaded = 0

        for mp3_file in tqdm(mp3_files, desc="Uploading"):
            remote_path = f"{onedrive_path.rstrip('/')}/{mp3_file.name}"

            if storage.upload(mp3_file, remote_path):
                uploaded += 1
                mp3_file.unlink()  # Delete after upload
            else:
                logger.error(f"Failed to upload: {mp3_file.name}")

        print(f"Uploaded {uploaded} files to OneDrive")

    # Cleanup
    if not keep_zip and zip_path.exists():
        print(f"Removing zip file: {zip_path}")
        zip_path.unlink()

    print("\nDone!")


def list_subsets():
    """List available FMA subsets"""
    print("Free Music Archive (FMA) Dataset")
    print("=" * 60)
    print("Website: https://github.com/mdeff/fma")
    print()

    for name, info in FMA_SUBSETS.items():
        print(f"{name}:")
        print(f"  Tracks: {info['tracks']:,}")
        print(f"  Size: {info['size_gb']} GB")
        print(f"  Description: {info['description']}")
        print()

    print("Recommended for ML training:")
    print("  - 'small' for testing (7.2 GB, 8K tracks)")
    print("  - 'medium' for production training (22 GB, 25K tracks)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FMA Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available subsets
  python download_fma.py list

  # Download small subset locally
  python download_fma.py download --subset small --dest ./data/fma

  # Download medium subset to OneDrive
  python download_fma.py download --subset medium --dest onedrive://FMA/

  # Download with file limit
  python download_fma.py download --subset large --dest ./data --max 1000
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # List command
    subparsers.add_parser("list", help="List available subsets")

    # Download command
    dl = subparsers.add_parser("download", help="Download FMA subset")
    dl.add_argument("--subset", "-s", default="small",
                    choices=list(FMA_SUBSETS.keys()),
                    help="Dataset subset to download")
    dl.add_argument("--dest", "-d", required=True,
                    help="Destination (local path or onedrive://path)")
    dl.add_argument("--max", "-n", type=int, default=0,
                    help="Max files to extract (0 = all)")
    dl.add_argument("--keep-zip", action="store_true",
                    help="Keep zip file after extraction")
    dl.add_argument("--no-skip", action="store_true",
                    help="Don't skip existing files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "list":
        list_subsets()

    elif args.command == "download":
        download_fma(
            subset=args.subset,
            dest=args.dest,
            max_files=args.max,
            keep_zip=args.keep_zip,
            skip_existing=not args.no_skip
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
