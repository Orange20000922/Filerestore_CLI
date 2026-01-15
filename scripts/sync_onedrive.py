#!/usr/bin/env python3
"""
OneDrive Sync Tool

Bidirectional sync between local filesystem and OneDrive for Business.

Usage:
    # Pull files from OneDrive to local
    python sync_onedrive.py pull --source Govdocs/ --dest ./data/govdocs --types zip,docx

    # Push files from local to OneDrive
    python sync_onedrive.py push --source ./output/dataset.csv --dest ML/datasets/

    # List remote files
    python sync_onedrive.py list --path Govdocs/ --types zip
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from storage import get_storage, OneDriveBackend

logger = logging.getLogger(__name__)


def parse_types(types_str: str) -> Set[str]:
    """Parse comma-separated file types"""
    if not types_str:
        return set()
    return {t.strip().lower().lstrip('.') for t in types_str.split(',')}


def matches_types(filename: str, types: Set[str]) -> bool:
    """Check if filename matches any of the specified types"""
    if not types:
        return True
    ext = Path(filename).suffix.lower().lstrip('.')
    return ext in types


def pull_files(
    source: str,
    dest: str,
    types: Set[str] = None,
    max_files: int = 0,
    max_size_mb: int = 0,
    skip_existing: bool = True,
    dry_run: bool = False
):
    """
    Pull files from OneDrive to local

    Args:
        source: OneDrive source path (e.g., "Govdocs/")
        dest: Local destination directory
        types: Set of file extensions to download
        max_files: Maximum number of files to download (0 = unlimited)
        max_size_mb: Maximum file size in MB (0 = unlimited)
        skip_existing: Skip files that already exist locally
        dry_run: Show what would be done without downloading
    """
    storage = get_storage("onedrive")

    print(f"Listing files in OneDrive:{source}...")
    all_files = storage.list_files(source)

    if not all_files:
        print("No files found or authentication failed.")
        return

    # Filter by type
    if types:
        all_files = [f for f in all_files if matches_types(f, types)]

    print(f"Found {len(all_files)} files matching criteria")

    # Apply max_files limit
    if max_files > 0:
        all_files = all_files[:max_files]

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0

    for remote_file in tqdm(all_files, desc="Downloading"):
        # Compute local path
        rel_path = remote_file
        if source and remote_file.startswith(source):
            rel_path = remote_file[len(source):].lstrip('/')

        local_file = dest_path / rel_path

        # Check if exists
        if skip_existing and local_file.exists():
            skipped += 1
            continue

        # Check file size
        if max_size_mb > 0:
            info = storage.get_file_info(remote_file)
            if info and info.get("size", 0) > max_size_mb * 1024 * 1024:
                skipped += 1
                continue

        if dry_run:
            print(f"Would download: {remote_file} -> {local_file}")
            downloaded += 1
            continue

        # Download
        local_file.parent.mkdir(parents=True, exist_ok=True)

        if storage.download(remote_file, local_file):
            downloaded += 1
        else:
            errors += 1
            logger.error(f"Failed to download: {remote_file}")

    print(f"\nDownloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")


def push_files(
    source: str,
    dest: str,
    types: Set[str] = None,
    overwrite: bool = False,
    dry_run: bool = False
):
    """
    Push files from local to OneDrive

    Args:
        source: Local source path (file or directory)
        dest: OneDrive destination path
        types: Set of file extensions to upload
        overwrite: Overwrite existing files
        dry_run: Show what would be done without uploading
    """
    storage = get_storage("onedrive")
    source_path = Path(source)

    if not source_path.exists():
        print(f"Source not found: {source}")
        return

    # Collect files to upload
    if source_path.is_file():
        files_to_upload = [(source_path, dest)]
    else:
        files_to_upload = []
        for local_file in source_path.rglob("*"):
            if not local_file.is_file():
                continue

            if types and not matches_types(local_file.name, types):
                continue

            rel_path = local_file.relative_to(source_path)
            remote_path = f"{dest.rstrip('/')}/{rel_path}".replace("\\", "/")
            files_to_upload.append((local_file, remote_path))

    print(f"Found {len(files_to_upload)} files to upload")

    uploaded = 0
    skipped = 0
    errors = 0

    for local_file, remote_path in tqdm(files_to_upload, desc="Uploading"):
        # Check if exists
        if not overwrite and storage.exists(remote_path):
            skipped += 1
            continue

        if dry_run:
            print(f"Would upload: {local_file} -> OneDrive:{remote_path}")
            uploaded += 1
            continue

        if storage.upload(local_file, remote_path):
            uploaded += 1
        else:
            errors += 1
            logger.error(f"Failed to upload: {local_file}")

    print(f"\nUploaded: {uploaded}, Skipped: {skipped}, Errors: {errors}")


def list_files(
    path: str,
    types: Set[str] = None,
    show_size: bool = False
):
    """List files in OneDrive"""
    storage = get_storage("onedrive")

    print(f"Listing OneDrive:{path}...")
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
            size_str = format_size(size)
            print(f"{size_str:>10}  {f}")
        else:
            print(f)

    print(f"\nTotal: {len(files)} files")


def format_size(size: int) -> str:
    """Format size in bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="OneDrive Sync Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pull ZIP files from OneDrive
  python sync_onedrive.py pull --source Govdocs/ --dest ./data --types zip --max-files 100

  # Push CSV to OneDrive
  python sync_onedrive.py push --source ./dataset.csv --dest ML/datasets/

  # List files
  python sync_onedrive.py list --path Govdocs/ --types zip --size
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Download files from OneDrive")
    pull_parser.add_argument("--source", "-s", required=True, help="OneDrive source path")
    pull_parser.add_argument("--dest", "-d", required=True, help="Local destination directory")
    pull_parser.add_argument("--types", "-t", help="File types to download (comma-separated)")
    pull_parser.add_argument("--max-files", "-n", type=int, default=0, help="Max files to download")
    pull_parser.add_argument("--max-size", type=int, default=0, help="Max file size in MB")
    pull_parser.add_argument("--no-skip", action="store_true", help="Don't skip existing files")
    pull_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    # Push command
    push_parser = subparsers.add_parser("push", help="Upload files to OneDrive")
    push_parser.add_argument("--source", "-s", required=True, help="Local source path")
    push_parser.add_argument("--dest", "-d", required=True, help="OneDrive destination path")
    push_parser.add_argument("--types", "-t", help="File types to upload (comma-separated)")
    push_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    push_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    # List command
    list_parser = subparsers.add_parser("list", help="List files in OneDrive")
    list_parser.add_argument("--path", "-p", default="", help="OneDrive path")
    list_parser.add_argument("--types", "-t", help="File types to list (comma-separated)")
    list_parser.add_argument("--size", action="store_true", help="Show file sizes")

    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Authenticate with OneDrive")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "pull":
        types = parse_types(args.types) if args.types else None
        pull_files(
            source=args.source,
            dest=args.dest,
            types=types,
            max_files=args.max_files,
            max_size_mb=args.max_size,
            skip_existing=not args.no_skip,
            dry_run=args.dry_run
        )

    elif args.command == "push":
        types = parse_types(args.types) if args.types else None
        push_files(
            source=args.source,
            dest=args.dest,
            types=types,
            overwrite=args.overwrite,
            dry_run=args.dry_run
        )

    elif args.command == "list":
        types = parse_types(args.types) if args.types else None
        list_files(
            path=args.path,
            types=types,
            show_size=args.size
        )

    elif args.command == "auth":
        storage = get_storage("onedrive")
        if storage.authenticate():
            print("Authentication successful!")
        else:
            print("Authentication failed.")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
