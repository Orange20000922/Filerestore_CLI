#!/usr/bin/env python3
"""
Google Drive Sync Tool

Sync files between local filesystem and Google Drive.
Optimized for use with Google Colab.

Usage:
    # Upload dataset to Google Drive
    python sync_gdrive.py upload ./dataset.csv --dest ML/datasets/

    # Download model from Google Drive
    python sync_gdrive.py download ML/models/model.onnx --dest ./models/

    # List files
    python sync_gdrive.py list --path ML/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set
from tqdm import tqdm
import logging

sys.path.insert(0, str(Path(__file__).parent))
from storage import get_storage

logger = logging.getLogger(__name__)


def parse_types(types_str: str) -> Set[str]:
    """Parse comma-separated file types"""
    if not types_str:
        return set()
    return {t.strip().lower().lstrip('.') for t in types_str.split(',')}


def matches_types(filename: str, types: Set[str]) -> bool:
    """Check if filename matches types"""
    if not types:
        return True
    ext = Path(filename).suffix.lower().lstrip('.')
    return ext in types


def format_size(size: int) -> str:
    """Format size in bytes"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def upload_files(
    source: str,
    dest: str,
    types: Set[str] = None,
    overwrite: bool = False,
    dry_run: bool = False
):
    """Upload files to Google Drive"""
    storage = get_storage("gdrive")
    source_path = Path(source)

    if not source_path.exists():
        print(f"Source not found: {source}")
        return

    # Collect files
    if source_path.is_file():
        files = [(source_path, f"{dest.rstrip('/')}/{source_path.name}")]
    else:
        files = []
        for f in source_path.rglob("*"):
            if f.is_file():
                if types and not matches_types(f.name, types):
                    continue
                rel = f.relative_to(source_path)
                remote = f"{dest.rstrip('/')}/{rel}".replace("\\", "/")
                files.append((f, remote))

    print(f"Found {len(files)} files to upload")

    uploaded = 0
    skipped = 0
    errors = 0

    for local_file, remote_path in tqdm(files, desc="Uploading"):
        if not overwrite and storage.exists(remote_path):
            skipped += 1
            continue

        if dry_run:
            print(f"Would upload: {local_file} -> GDrive:{remote_path}")
            uploaded += 1
            continue

        if storage.upload(local_file, remote_path):
            uploaded += 1
        else:
            errors += 1

    print(f"\nUploaded: {uploaded}, Skipped: {skipped}, Errors: {errors}")


def download_files(
    source: str,
    dest: str,
    types: Set[str] = None,
    skip_existing: bool = True,
    dry_run: bool = False
):
    """Download files from Google Drive"""
    storage = get_storage("gdrive")
    dest_path = Path(dest)

    # Check if source is a file or directory
    info = storage.get_file_info(source)

    if info and info.get("is_file"):
        # Single file download
        files = [(source, dest_path / Path(source).name)]
    else:
        # Directory download
        all_files = storage.list_files(source)
        if types:
            all_files = [f for f in all_files if matches_types(f, types)]

        files = []
        for remote in all_files:
            rel = remote
            if source and remote.startswith(source):
                rel = remote[len(source):].lstrip('/')
            local = dest_path / rel
            files.append((remote, local))

    print(f"Found {len(files)} files to download")

    downloaded = 0
    skipped = 0
    errors = 0

    for remote_path, local_path in tqdm(files, desc="Downloading"):
        if skip_existing and local_path.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"Would download: GDrive:{remote_path} -> {local_path}")
            downloaded += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        if storage.download(remote_path, local_path):
            downloaded += 1
        else:
            errors += 1

    print(f"\nDownloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")


def list_files(path: str, types: Set[str] = None, show_size: bool = False):
    """List files in Google Drive"""
    storage = get_storage("gdrive")

    print(f"Listing GDrive:{path}...")
    files = storage.list_files(path)

    if types:
        files = [f for f in files if matches_types(f, types)]

    if not files:
        print("No files found.")
        return

    for f in files:
        if show_size:
            info = storage.get_file_info(f)
            size = info.get("size", 0) if info else 0
            print(f"{format_size(size):>10}  {f}")
        else:
            print(f)

    print(f"\nTotal: {len(files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Google Drive Sync Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload dataset
  python sync_gdrive.py upload ./dataset.csv --dest ML/datasets/

  # Download model
  python sync_gdrive.py download ML/models/model.onnx --dest ./models/

  # List files
  python sync_gdrive.py list --path ML/ --types onnx,pt
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # Upload
    up = subparsers.add_parser("upload", help="Upload to Google Drive")
    up.add_argument("source", help="Local source path")
    up.add_argument("--dest", "-d", required=True, help="GDrive destination")
    up.add_argument("--types", "-t", help="File types")
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--dry-run", action="store_true")

    # Download
    dl = subparsers.add_parser("download", help="Download from Google Drive")
    dl.add_argument("source", help="GDrive source path")
    dl.add_argument("--dest", "-d", required=True, help="Local destination")
    dl.add_argument("--types", "-t", help="File types")
    dl.add_argument("--no-skip", action="store_true")
    dl.add_argument("--dry-run", action="store_true")

    # List
    ls = subparsers.add_parser("list", help="List files")
    ls.add_argument("--path", "-p", default="", help="GDrive path")
    ls.add_argument("--types", "-t", help="File types")
    ls.add_argument("--size", action="store_true")

    # Auth
    subparsers.add_parser("auth", help="Authenticate")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "upload":
        types = parse_types(args.types) if args.types else None
        upload_files(args.source, args.dest, types, args.overwrite, args.dry_run)

    elif args.command == "download":
        types = parse_types(args.types) if args.types else None
        download_files(args.source, args.dest, types, not args.no_skip, args.dry_run)

    elif args.command == "list":
        types = parse_types(args.types) if args.types else None
        list_files(args.path, types, args.size)

    elif args.command == "auth":
        storage = get_storage("gdrive")
        if storage.authenticate():
            print("Authentication successful!")
        else:
            print("Authentication failed.")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
