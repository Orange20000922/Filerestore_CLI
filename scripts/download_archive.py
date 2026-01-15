#!/usr/bin/env python3
"""
Internet Archive Media Downloader

Downloads multimedia files from the Internet Archive for ML training.
The Internet Archive contains millions of free media files in various formats.

Collections:
- audio: MP3, WAV, FLAC audio files
- movies: MP4, AVI, MKV video files
- images: JPG, PNG image files

Usage:
    # List available collections
    python download_archive.py list

    # Download audio files
    python download_archive.py download --collection audio --count 500 --dest ./data/archive

    # Download video files
    python download_archive.py download --collection movies --count 100 --dest ./data/archive

    # Download to OneDrive
    python download_archive.py download --collection audio --count 100 --dest onedrive://Archive/
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set
import requests
from tqdm import tqdm
import logging
import time
import json

sys.path.insert(0, str(Path(__file__).parent))
from storage import get_storage

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

ARCHIVE_API_BASE = "https://archive.org"
ARCHIVE_SEARCH_API = f"{ARCHIVE_API_BASE}/advancedsearch.php"
ARCHIVE_METADATA_API = f"{ARCHIVE_API_BASE}/metadata"
ARCHIVE_DOWNLOAD_BASE = f"{ARCHIVE_API_BASE}/download"

# Media type configurations
MEDIA_COLLECTIONS = {
    "audio": {
        "mediatype": "audio",
        "formats": {"mp3", "wav", "flac", "ogg"},
        "query": "mediatype:audio AND format:(MP3 OR WAV OR FLAC)",
        "description": "Audio files (MP3, WAV, FLAC)"
    },
    "movies": {
        "mediatype": "movies",
        "formats": {"mp4", "avi", "mkv", "mov", "webm"},
        "query": "mediatype:movies AND format:(MPEG4 OR h.264)",
        "description": "Video files (MP4, AVI, MKV)"
    },
    "images": {
        "mediatype": "image",
        "formats": {"jpg", "jpeg", "png", "gif", "tiff"},
        "query": "mediatype:image AND format:(JPEG OR PNG)",
        "description": "Image files (JPG, PNG)"
    },
    "software": {
        "mediatype": "software",
        "formats": {"zip", "exe", "iso", "dmg"},
        "query": "mediatype:software AND format:(ZIP OR ISO)",
        "description": "Software archives (ZIP, ISO)"
    }
}

# File size limits (bytes)
MIN_FILE_SIZE = 50 * 1024  # 50 KB minimum
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB maximum


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Internet Archive API
# =============================================================================

class ArchiveAPI:
    """Internet Archive API client"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FileRestore-ML-DataCollector/1.0"
        })

    def search(
        self,
        query: str,
        rows: int = 100,
        page: int = 1,
        sort: str = "downloads desc"
    ) -> Dict:
        """
        Search Internet Archive

        Args:
            query: Search query
            rows: Results per page (max 10000)
            page: Page number
            sort: Sort order

        Returns:
            Search results
        """
        params = {
            "q": query,
            "fl[]": ["identifier", "title", "mediatype", "downloads", "item_size"],
            "rows": min(rows, 10000),
            "page": page,
            "sort[]": sort,
            "output": "json"
        }

        response = self.session.get(ARCHIVE_SEARCH_API, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_item_files(self, identifier: str) -> List[Dict]:
        """
        Get files for an item

        Args:
            identifier: Item identifier

        Returns:
            List of file metadata
        """
        url = f"{ARCHIVE_METADATA_API}/{identifier}/files"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])

    def get_download_url(self, identifier: str, filename: str) -> str:
        """Get download URL for a file"""
        return f"{ARCHIVE_DOWNLOAD_BASE}/{identifier}/{filename}"


# =============================================================================
# Download Functions
# =============================================================================

def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 1024 * 1024
) -> bool:
    """Download a file"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Skip if too large
        if total_size > MAX_FILE_SIZE:
            logger.debug(f"Skipping {dest_path.name}: too large ({format_size(total_size)})")
            return False

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix('.tmp')

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        temp_path.rename(dest_path)
        return True

    except Exception as e:
        logger.debug(f"Download failed: {e}")
        return False


def collect_files_from_items(
    api: ArchiveAPI,
    items: List[Dict],
    formats: Set[str],
    max_files: int
) -> List[Dict]:
    """
    Collect downloadable files from items

    Args:
        api: Archive API client
        items: List of items
        formats: Allowed file formats
        max_files: Maximum files to collect

    Returns:
        List of file info dicts
    """
    files = []

    for item in tqdm(items, desc="Scanning items"):
        if len(files) >= max_files:
            break

        identifier = item.get("identifier")
        if not identifier:
            continue

        try:
            item_files = api.get_item_files(identifier)

            for f in item_files:
                if len(files) >= max_files:
                    break

                name = f.get("name", "")
                size = int(f.get("size", 0))
                ext = Path(name).suffix.lower().lstrip('.')

                # Check format and size
                if ext not in formats:
                    continue
                if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                    continue

                files.append({
                    "identifier": identifier,
                    "filename": name,
                    "size": size,
                    "url": api.get_download_url(identifier, name)
                })

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            logger.debug(f"Failed to get files for {identifier}: {e}")

    return files


def download_archive_media(
    collection: str,
    count: int,
    dest: str,
    skip_existing: bool = True
):
    """
    Download media from Internet Archive

    Args:
        collection: Media collection (audio, movies, images)
        count: Number of files to download
        dest: Destination path
        skip_existing: Skip existing files
    """
    if collection not in MEDIA_COLLECTIONS:
        print(f"Error: Unknown collection '{collection}'")
        print(f"Available: {list(MEDIA_COLLECTIONS.keys())}")
        return

    config = MEDIA_COLLECTIONS[collection]
    print(f"\nInternet Archive - {config['description']}")
    print(f"Formats: {', '.join(config['formats'])}")

    # Determine storage backend
    if dest.startswith("onedrive://"):
        upload_to_onedrive = True
        onedrive_path = dest[11:]
        local_dest = Path(tempfile.gettempdir()) / "archive_temp"
        storage = get_storage("onedrive")
    else:
        upload_to_onedrive = False
        local_dest = Path(dest) / collection
        storage = None

    local_dest.mkdir(parents=True, exist_ok=True)

    api = ArchiveAPI()

    # Search for items
    print(f"\nSearching for items...")
    items = []
    page = 1
    items_needed = count * 3  # Overfetch to account for filtering

    while len(items) < items_needed:
        try:
            result = api.search(
                query=config["query"],
                rows=100,
                page=page
            )

            docs = result.get("response", {}).get("docs", [])
            if not docs:
                break

            items.extend(docs)
            page += 1
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Search error: {e}")
            break

    print(f"Found {len(items)} items")

    if not items:
        print("No items found!")
        return

    # Collect files
    print(f"\nCollecting file information...")
    files = collect_files_from_items(api, items, config["formats"], count)
    print(f"Found {len(files)} downloadable files")

    if not files:
        print("No suitable files found!")
        return

    # Download files
    downloaded = 0
    skipped = 0
    errors = 0

    print(f"\nDownloading files...")
    for file_info in tqdm(files, desc="Downloading"):
        filename = file_info["filename"]
        url = file_info["url"]

        # Create unique filename
        safe_filename = f"{file_info['identifier']}_{filename}"
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in '._-')
        local_path = local_dest / safe_filename

        # Check existing
        if skip_existing and local_path.exists():
            skipped += 1
            continue

        if upload_to_onedrive and storage:
            remote_path = f"{onedrive_path.rstrip('/')}/{collection}/{safe_filename}"
            if skip_existing and storage.exists(remote_path):
                skipped += 1
                continue

        # Download
        if download_file(url, local_path):
            downloaded += 1

            # Upload to OneDrive
            if upload_to_onedrive and storage:
                remote_path = f"{onedrive_path.rstrip('/')}/{collection}/{safe_filename}"
                if storage.upload(local_path, remote_path):
                    local_path.unlink()
                else:
                    errors += 1
        else:
            errors += 1

        # Rate limiting
        time.sleep(0.2)

    print(f"\nDownloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")


def list_collections():
    """List available media collections"""
    print("Internet Archive Media Collections")
    print("=" * 60)
    print("Website: https://archive.org")
    print()

    for name, config in MEDIA_COLLECTIONS.items():
        print(f"{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Formats: {', '.join(config['formats'])}")
        print()

    print("Notes:")
    print(f"  - Minimum file size: {format_size(MIN_FILE_SIZE)}")
    print(f"  - Maximum file size: {format_size(MAX_FILE_SIZE)}")
    print("  - All content is public domain or freely licensed")


def download_all_media(count_per_type: int, dest: str, **kwargs):
    """Download media from all collections"""
    collections = ["audio", "movies"]  # Main media types

    print(f"Downloading {count_per_type} files from each collection:")
    print(f"Collections: {', '.join(collections)}")

    for collection in collections:
        print(f"\n{'='*60}")
        print(f"Collection: {collection}")
        print("=" * 60)

        download_archive_media(
            collection=collection,
            count=count_per_type,
            dest=dest,
            **kwargs
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Internet Archive Media Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available collections
  python download_archive.py list

  # Download 500 audio files
  python download_archive.py download --collection audio --count 500 --dest ./data/archive

  # Download 100 video files
  python download_archive.py download --collection movies --count 100 --dest ./data/archive

  # Download from all media collections
  python download_archive.py download --collection all --count 200 --dest ./data/archive

  # Download to OneDrive
  python download_archive.py download --collection audio --count 100 --dest onedrive://Archive/
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # List command
    subparsers.add_parser("list", help="List available collections")

    # Download command
    dl = subparsers.add_parser("download", help="Download media files")
    dl.add_argument("--collection", "-c", default="audio",
                    choices=list(MEDIA_COLLECTIONS.keys()) + ["all"],
                    help="Media collection to download from")
    dl.add_argument("--count", "-n", type=int, default=100,
                    help="Number of files to download")
    dl.add_argument("--dest", "-d", required=True,
                    help="Destination (local path or onedrive://path)")
    dl.add_argument("--no-skip", action="store_true",
                    help="Don't skip existing files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "list":
        list_collections()

    elif args.command == "download":
        if args.collection == "all":
            download_all_media(
                count_per_type=args.count,
                dest=args.dest,
                skip_existing=not args.no_skip
            )
        else:
            download_archive_media(
                collection=args.collection,
                count=args.count,
                dest=args.dest,
                skip_existing=not args.no_skip
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
