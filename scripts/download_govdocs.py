#!/usr/bin/env python3
"""
Govdocs Dataset Downloader

Downloads ZIP archives from Digital Corpora's Govdocs1 dataset.
Can download directly to local storage or upload to OneDrive.

Govdocs1 contains ~1 million files in various formats (PDF, DOC, XLS, ZIP, etc.)
organized into 1000 ZIP archives (govdocs1-000.zip to govdocs1-999.zip).

Usage:
    # List available archives
    python download_govdocs.py list

    # Download specific range to local
    python download_govdocs.py download --range 001-010 --dest ./data/govdocs

    # Download and upload to OneDrive
    python download_govdocs.py download --range 001-010 --dest onedrive://Govdocs/

    # Download specific archives
    python download_govdocs.py download --archives 001,005,010 --dest ./data
"""

import argparse
import os
import sys
import time
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from storage import get_storage

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Updated URL - Digital Corpora moved to S3
# Old URL: https://digitalcorpora.org/downloads/govdocs1/
GOVDOCS_BASE_URL = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/"
# Alternative URL patterns to try if primary fails
GOVDOCS_ALT_URLS = [
    "https://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/",
    "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/",
]
# Archive naming patterns: try both old and new formats
GOVDOCS_ARCHIVE_PATTERNS = [
    "{:03d}.zip",           # New naming: 000.zip, 001.zip
    "govdocs1-{:03d}.zip",  # Old naming: govdocs1-000.zip
]
GOVDOCS_TOTAL_ARCHIVES = 1000  # 000-999

# Each archive is approximately 450-550 MB
EXPECTED_ARCHIVE_SIZE_MB = 500


def get_archive_url(archive_num: int) -> str:
    """Get download URL for archive number (primary URL)"""
    filename = GOVDOCS_ARCHIVE_PATTERNS[0].format(archive_num)
    return f"{GOVDOCS_BASE_URL}{filename}"


def get_archive_urls(archive_num: int) -> List[str]:
    """Get all possible download URLs for archive number"""
    urls = []
    for base_url in GOVDOCS_ALT_URLS:
        for pattern in GOVDOCS_ARCHIVE_PATTERNS:
            filename = pattern.format(archive_num)
            urls.append(f"{base_url}{filename}")
    return urls


def parse_range(range_str: str) -> List[int]:
    """Parse range string like '001-010' or '1-10' to list of integers"""
    parts = range_str.split('-')
    if len(parts) == 2:
        start = int(parts[0])
        end = int(parts[1])
        return list(range(start, end + 1))
    else:
        return [int(parts[0])]


def parse_archives(archives_str: str) -> List[int]:
    """Parse comma-separated archive numbers"""
    return [int(x.strip()) for x in archives_str.split(',')]


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_time(seconds: float) -> str:
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# Download Functions
# =============================================================================

def check_archive_exists(archive_num: int) -> Tuple[bool, int, str]:
    """Check if archive exists and get its size. Returns (exists, size, working_url)"""
    # Try all possible URLs
    for url in get_archive_urls(archive_num):
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                return True, size, url
        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            continue
    return False, 0, ""


def download_archive(
    archive_num: int,
    dest_path: Path,
    progress_callback: Optional[callable] = None,
    chunk_size: int = 1024 * 1024  # 1MB chunks
) -> bool:
    """
    Download a single archive

    Args:
        archive_num: Archive number (0-999)
        dest_path: Local destination path
        progress_callback: Optional callback(downloaded, total)
        chunk_size: Download chunk size

    Returns:
        True if successful
    """
    # Try all possible URLs until one works
    urls = get_archive_urls(archive_num)
    last_error = None
    filename = f"{archive_num:03d}.zip"

    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Download to temp file first
            temp_path = dest_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # Rename to final path
            temp_path.rename(dest_path)

            logger.info(f"Downloaded {filename} from {url} ({format_size(total_size)})")
            return True

        except Exception as e:
            last_error = e
            logger.debug(f"Failed to download from {url}: {e}")
            continue

    logger.error(f"Failed to download archive {archive_num}: {last_error}")
    return False


def download_archives(
    archive_nums: List[int],
    dest: str,
    workers: int = 2,
    skip_existing: bool = True,
    upload_to_onedrive: bool = False
):
    """
    Download multiple archives

    Args:
        archive_nums: List of archive numbers to download
        dest: Destination path (local or onedrive://path)
        workers: Number of parallel downloads
        skip_existing: Skip already downloaded files
        upload_to_onedrive: Upload to OneDrive after download
    """
    # Determine if uploading to OneDrive
    if dest.startswith("onedrive://"):
        upload_to_onedrive = True
        onedrive_path = dest[11:]  # Remove "onedrive://"
        local_dest = Path(tempfile.gettempdir()) / "govdocs_temp"
        storage = get_storage("onedrive")
    else:
        local_dest = Path(dest)
        storage = None

    local_dest.mkdir(parents=True, exist_ok=True)

    # Filter out existing files
    to_download = []
    for num in archive_nums:
        filename = f"{num:03d}.zip"  # Local naming convention
        local_path = local_dest / filename

        if upload_to_onedrive:
            remote_path = f"{onedrive_path.rstrip('/')}/{filename}"
            if skip_existing and storage.exists(remote_path):
                logger.info(f"Skipping {filename} (exists on OneDrive)")
                continue
        elif skip_existing and local_path.exists():
            logger.info(f"Skipping {filename} (exists locally)")
            continue

        to_download.append(num)

    if not to_download:
        print("All archives already downloaded.")
        return

    print(f"\nDownloading {len(to_download)} archives...")
    total_expected_size = len(to_download) * EXPECTED_ARCHIVE_SIZE_MB * 1024 * 1024
    print(f"Estimated total size: {format_size(total_expected_size)}")

    # Progress tracking
    success = 0
    failed = 0

    # Download with progress bar
    with tqdm(total=len(to_download), desc="Archives", unit="file") as pbar:
        for num in to_download:
            filename = f"{num:03d}.zip"  # Local naming convention
            local_path = local_dest / filename

            # Create per-file progress bar
            with tqdm(desc=filename, unit='B', unit_scale=True, leave=False) as file_pbar:
                def update_progress(downloaded, total):
                    file_pbar.total = total
                    file_pbar.n = downloaded
                    file_pbar.refresh()

                if download_archive(num, local_path, progress_callback=update_progress):
                    success += 1

                    # Upload to OneDrive if needed
                    if upload_to_onedrive and storage:
                        remote_path = f"{onedrive_path.rstrip('/')}/{filename}"
                        print(f"\nUploading {filename} to OneDrive...")

                        if storage.upload(local_path, remote_path):
                            # Delete local temp file after upload
                            local_path.unlink()
                            logger.info(f"Uploaded and cleaned up {filename}")
                        else:
                            logger.error(f"Failed to upload {filename}")
                else:
                    failed += 1

            pbar.update(1)

    print(f"\n\nDownload complete!")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

    if upload_to_onedrive:
        print(f"  Uploaded to: OneDrive:{onedrive_path}")
    else:
        print(f"  Saved to: {local_dest}")


def list_archives(check_remote: bool = False):
    """List available Govdocs archives"""
    print("Govdocs1 Dataset Archives")
    print("=" * 50)
    print(f"Total archives: {GOVDOCS_TOTAL_ARCHIVES} (000-999)")
    print(f"Expected size per archive: ~{EXPECTED_ARCHIVE_SIZE_MB} MB")
    print(f"Total dataset size: ~{GOVDOCS_TOTAL_ARCHIVES * EXPECTED_ARCHIVE_SIZE_MB / 1024:.0f} GB")
    print()
    print("Archive format: XXX.zip (000.zip to 999.zip)")
    print(f"Base URL: {GOVDOCS_BASE_URL}")
    print()

    if check_remote:
        print("Checking archive availability (first 10)...")
        for i in range(10):
            exists, size, url = check_archive_exists(i)
            status = f"{format_size(size)}" if exists else "NOT FOUND"
            filename = f"{i:03d}.zip"
            print(f"  {filename}: {status}")
            if exists and url:
                print(f"    URL: {url}")


def extract_specific_types(
    archive_path: Path,
    output_dir: Path,
    types: List[str],
    max_files: int = 0
) -> int:
    """
    Extract specific file types from a ZIP archive

    Args:
        archive_path: Path to ZIP archive
        output_dir: Output directory
        types: List of extensions to extract (e.g., ['zip', 'docx'])
        max_files: Maximum files to extract (0 = unlimited)

    Returns:
        Number of files extracted
    """
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)
    types_lower = {t.lower().lstrip('.') for t in types}

    extracted = 0

    with zipfile.ZipFile(archive_path, 'r') as zf:
        for member in zf.namelist():
            if max_files > 0 and extracted >= max_files:
                break

            ext = Path(member).suffix.lower().lstrip('.')
            if ext in types_lower:
                # Extract file
                zf.extract(member, output_dir)
                extracted += 1

    return extracted


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Govdocs Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available archives
  python download_govdocs.py list

  # Download archives 001-010 to local directory
  python download_govdocs.py download --range 001-010 --dest ./data/govdocs

  # Download and upload to OneDrive
  python download_govdocs.py download --range 001-005 --dest onedrive://Govdocs/

  # Download specific archives
  python download_govdocs.py download --archives 001,050,100 --dest ./data

  # Extract ZIP files from downloaded archive
  python download_govdocs.py extract ./govdocs1-001.zip --types zip,docx --dest ./extracted
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available archives")
    list_parser.add_argument("--check", action="store_true", help="Check remote availability")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download archives")
    dl_parser.add_argument("--range", "-r", help="Archive range (e.g., 001-010)")
    dl_parser.add_argument("--archives", "-a", help="Specific archives (e.g., 001,005,010)")
    dl_parser.add_argument("--dest", "-d", required=True, help="Destination (local path or onedrive://path)")
    dl_parser.add_argument("--workers", "-w", type=int, default=2, help="Parallel downloads")
    dl_parser.add_argument("--no-skip", action="store_true", help="Don't skip existing files")

    # Extract command
    ext_parser = subparsers.add_parser("extract", help="Extract files from archive")
    ext_parser.add_argument("archive", help="Archive path")
    ext_parser.add_argument("--types", "-t", required=True, help="File types to extract")
    ext_parser.add_argument("--dest", "-d", required=True, help="Output directory")
    ext_parser.add_argument("--max", "-n", type=int, default=0, help="Max files to extract")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "list":
        list_archives(check_remote=args.check)

    elif args.command == "download":
        if args.range:
            archive_nums = parse_range(args.range)
        elif args.archives:
            archive_nums = parse_archives(args.archives)
        else:
            print("Error: Specify --range or --archives")
            sys.exit(1)

        # Validate range
        invalid = [n for n in archive_nums if n < 0 or n >= GOVDOCS_TOTAL_ARCHIVES]
        if invalid:
            print(f"Error: Invalid archive numbers: {invalid}")
            sys.exit(1)

        download_archives(
            archive_nums=archive_nums,
            dest=args.dest,
            workers=args.workers,
            skip_existing=not args.no_skip
        )

    elif args.command == "extract":
        archive_path = Path(args.archive)
        if not archive_path.exists():
            print(f"Error: Archive not found: {archive_path}")
            sys.exit(1)

        types = [t.strip() for t in args.types.split(',')]
        output_dir = Path(args.dest)

        print(f"Extracting {types} from {archive_path}...")
        extracted = extract_specific_types(
            archive_path, output_dir, types, args.max
        )
        print(f"Extracted {extracted} files to {output_dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
