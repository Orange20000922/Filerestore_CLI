#!/usr/bin/env python3
"""
Pexels Video Downloader

Downloads MP4 video files from Pexels for ML training.
Pexels provides free high-quality stock videos via their API.

Features:
- Free API (requires API key from pexels.com)
- High-quality MP4 videos
- Various categories and search queries
- Multiple quality options (HD, SD)

Usage:
    # Set API key first
    export PEXELS_API_KEY="your_api_key_here"

    # Download popular videos
    python download_pexels.py download --query popular --count 100 --dest ./data/pexels

    # Download nature videos
    python download_pexels.py download --query nature --count 50 --dest ./data/pexels

    # Download to OneDrive
    python download_pexels.py download --query city --count 100 --dest onedrive://Pexels/
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
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

PEXELS_API_BASE = "https://api.pexels.com/videos"
PEXELS_API_KEY_ENV = "PEXELS_API_KEY"

# Popular search queries for diverse video content
DEFAULT_QUERIES = [
    "nature", "city", "people", "technology", "food",
    "travel", "animals", "sports", "business", "abstract"
]

# Quality preferences (in order of preference)
QUALITY_PREFERENCE = ["hd", "sd", "hls"]


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_api_key() -> Optional[str]:
    """Get Pexels API key from environment"""
    key = os.environ.get(PEXELS_API_KEY_ENV)
    if not key:
        # Try config file
        config_path = Path.home() / ".filerestore" / "pexels.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                key = config.get("api_key")
    return key


def save_api_key(api_key: str):
    """Save API key to config file"""
    config_dir = Path.home() / ".filerestore"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "pexels.json"
    with open(config_path, 'w') as f:
        json.dump({"api_key": api_key}, f)

    print(f"API key saved to {config_path}")


# =============================================================================
# Pexels API
# =============================================================================

class PexelsAPI:
    """Pexels Video API client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": api_key}

    def search_videos(
        self,
        query: str,
        per_page: int = 80,
        page: int = 1,
        min_duration: int = 5,
        max_duration: int = 60
    ) -> Dict:
        """
        Search for videos

        Args:
            query: Search query
            per_page: Results per page (max 80)
            page: Page number
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        """
        params = {
            "query": query,
            "per_page": min(per_page, 80),
            "page": page,
            "min_duration": min_duration,
            "max_duration": max_duration
        }

        response = requests.get(
            f"{PEXELS_API_BASE}/search",
            headers=self.headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_popular_videos(self, per_page: int = 80, page: int = 1) -> Dict:
        """Get popular videos"""
        params = {
            "per_page": min(per_page, 80),
            "page": page
        }

        response = requests.get(
            f"{PEXELS_API_BASE}/popular",
            headers=self.headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_video_download_url(self, video: Dict, quality: str = "hd") -> Optional[str]:
        """
        Get download URL for video

        Args:
            video: Video object from API response
            quality: Preferred quality (hd, sd, hls)

        Returns:
            Download URL or None
        """
        video_files = video.get("video_files", [])

        if not video_files:
            return None

        # Sort by quality preference
        def quality_score(vf):
            q = vf.get("quality", "").lower()
            try:
                return QUALITY_PREFERENCE.index(q)
            except ValueError:
                return 999

        video_files_sorted = sorted(video_files, key=quality_score)

        # Find best MP4 file
        for vf in video_files_sorted:
            if vf.get("file_type") == "video/mp4":
                return vf.get("link")

        # Fallback to any file
        return video_files_sorted[0].get("link") if video_files_sorted else None


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file with progress"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.with_suffix('.tmp')

        with open(temp_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        temp_path.rename(dest_path)
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_pexels_videos(
    query: str,
    count: int,
    dest: str,
    quality: str = "hd",
    min_duration: int = 5,
    max_duration: int = 60,
    skip_existing: bool = True
):
    """
    Download videos from Pexels

    Args:
        query: Search query (or "popular" for popular videos)
        count: Number of videos to download
        dest: Destination path (local or onedrive://path)
        quality: Video quality (hd, sd)
        min_duration: Minimum video duration
        max_duration: Maximum video duration
        skip_existing: Skip existing files
    """
    api_key = get_api_key()
    if not api_key:
        print("Error: Pexels API key not found!")
        print(f"Set {PEXELS_API_KEY_ENV} environment variable or run:")
        print("  python download_pexels.py auth --key YOUR_API_KEY")
        print("\nGet free API key at: https://www.pexels.com/api/")
        return

    api = PexelsAPI(api_key)

    # Determine storage backend
    if dest.startswith("onedrive://"):
        upload_to_onedrive = True
        onedrive_path = dest[11:]
        local_dest = Path(tempfile.gettempdir()) / "pexels_temp"
        storage = get_storage("onedrive")
    else:
        upload_to_onedrive = False
        local_dest = Path(dest)
        storage = None

    local_dest.mkdir(parents=True, exist_ok=True)

    # Collect videos
    print(f"Searching for '{query}' videos...")
    videos = []
    page = 1

    while len(videos) < count:
        try:
            if query.lower() == "popular":
                result = api.get_popular_videos(per_page=80, page=page)
            else:
                result = api.search_videos(
                    query=query,
                    per_page=80,
                    page=page,
                    min_duration=min_duration,
                    max_duration=max_duration
                )

            new_videos = result.get("videos", [])
            if not new_videos:
                break

            videos.extend(new_videos)
            page += 1

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"API error: {e}")
            break

    videos = videos[:count]
    print(f"Found {len(videos)} videos")

    if not videos:
        print("No videos found!")
        return

    # Download videos
    downloaded = 0
    skipped = 0
    errors = 0

    for video in tqdm(videos, desc="Downloading"):
        video_id = video.get("id")
        url = api.get_video_download_url(video, quality)

        if not url:
            errors += 1
            continue

        filename = f"pexels_{video_id}.mp4"
        local_path = local_dest / filename

        # Check existing
        if skip_existing and local_path.exists():
            skipped += 1
            continue

        if upload_to_onedrive and storage:
            remote_path = f"{onedrive_path.rstrip('/')}/{filename}"
            if skip_existing and storage.exists(remote_path):
                skipped += 1
                continue

        # Download
        if download_file(url, local_path):
            downloaded += 1

            # Upload to OneDrive
            if upload_to_onedrive and storage:
                remote_path = f"{onedrive_path.rstrip('/')}/{filename}"
                if storage.upload(local_path, remote_path):
                    local_path.unlink()
                else:
                    errors += 1
        else:
            errors += 1

        # Rate limiting
        time.sleep(0.2)

    print(f"\nDownloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")


def download_diverse_videos(
    count: int,
    dest: str,
    **kwargs
):
    """Download videos from multiple categories for diversity"""
    per_category = max(count // len(DEFAULT_QUERIES), 10)

    print(f"Downloading {per_category} videos from {len(DEFAULT_QUERIES)} categories")
    print(f"Categories: {', '.join(DEFAULT_QUERIES)}")
    print()

    for query in DEFAULT_QUERIES:
        print(f"\n=== Category: {query} ===")
        download_pexels_videos(
            query=query,
            count=per_category,
            dest=dest,
            **kwargs
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pexels Video Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save API key (get free key at pexels.com/api)
  python download_pexels.py auth --key YOUR_API_KEY

  # Download 100 nature videos
  python download_pexels.py download --query nature --count 100 --dest ./data/pexels

  # Download popular videos
  python download_pexels.py download --query popular --count 50 --dest ./data/pexels

  # Download diverse videos from multiple categories
  python download_pexels.py download --query diverse --count 500 --dest ./data/pexels

  # Download to OneDrive
  python download_pexels.py download --query city --count 100 --dest onedrive://Pexels/
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # Auth command
    auth = subparsers.add_parser("auth", help="Save Pexels API key")
    auth.add_argument("--key", "-k", required=True, help="Pexels API key")

    # Download command
    dl = subparsers.add_parser("download", help="Download videos")
    dl.add_argument("--query", "-q", default="popular",
                    help="Search query (or 'popular', 'diverse')")
    dl.add_argument("--count", "-n", type=int, default=100,
                    help="Number of videos to download")
    dl.add_argument("--dest", "-d", required=True,
                    help="Destination (local path or onedrive://path)")
    dl.add_argument("--quality", choices=["hd", "sd"], default="hd",
                    help="Video quality")
    dl.add_argument("--min-duration", type=int, default=5,
                    help="Minimum video duration (seconds)")
    dl.add_argument("--max-duration", type=int, default=60,
                    help="Maximum video duration (seconds)")
    dl.add_argument("--no-skip", action="store_true",
                    help="Don't skip existing files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "auth":
        save_api_key(args.key)

    elif args.command == "download":
        if args.query.lower() == "diverse":
            download_diverse_videos(
                count=args.count,
                dest=args.dest,
                quality=args.quality,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                skip_existing=not args.no_skip
            )
        else:
            download_pexels_videos(
                query=args.query,
                count=args.count,
                dest=args.dest,
                quality=args.quality,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                skip_existing=not args.no_skip
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
