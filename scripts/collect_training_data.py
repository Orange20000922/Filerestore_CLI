#!/usr/bin/env python3
"""
Unified Training Data Collection Script

Orchestrates downloading from multiple data sources and prepares
training data for the continuity detection model.

Data Sources:
- Govdocs (ZIP, DOCX, PDF files)
- Free Music Archive (MP3 audio)
- Pexels (MP4 video)
- Internet Archive (mixed media)

Usage:
    # Download from all sources
    python collect_training_data.py download --dest ./data

    # Download specific sources
    python collect_training_data.py download --sources govdocs,fma --dest ./data

    # Extract features (using Python extractor)
    python collect_training_data.py extract --input ./data --output dataset.csv

    # Full pipeline
    python collect_training_data.py pipeline --dest ./data --output dataset.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DATA_SOURCES = {
    "govdocs": {
        "script": "download_govdocs.py",
        "default_args": ["download", "--range", "001-010", "--dest", "{dest}/govdocs"],
        "file_types": ["zip", "docx", "xlsx", "pptx", "pdf"],
        "description": "Digital Corpora Govdocs1 (office documents)"
    },
    "fma": {
        "script": "download_fma.py",
        "default_args": ["download", "--subset", "small", "--dest", "{dest}/fma"],
        "file_types": ["mp3"],
        "description": "Free Music Archive (MP3 audio)"
    },
    "pexels": {
        "script": "download_pexels.py",
        "default_args": ["download", "--query", "diverse", "--count", "500", "--dest", "{dest}/pexels"],
        "file_types": ["mp4"],
        "description": "Pexels (MP4 video)"
    },
    "archive": {
        "script": "download_archive.py",
        "default_args": ["download", "--collection", "all", "--count", "200", "--dest", "{dest}/archive"],
        "file_types": ["mp3", "mp4", "wav", "avi"],
        "description": "Internet Archive (mixed media)"
    }
}

# Default data collection configuration
DEFAULT_CONFIG = {
    "govdocs_range": "001-010",      # 10 archives (~5GB)
    "fma_subset": "small",           # 8000 tracks (~7.2GB)
    "pexels_count": 500,             # 500 videos
    "archive_count": 200,            # 200 files per type
    "samples_per_file": 10,          # Continuity samples per file
    "min_file_size": 32768,          # 32KB minimum
}


def get_script_path(script_name: str) -> Path:
    """Get full path to a download script."""
    scripts_dir = Path(__file__).parent
    return scripts_dir / script_name


def run_download_script(source: str, dest: str, extra_args: List[str] = None) -> bool:
    """
    Run a download script.

    Args:
        source: Source name (govdocs, fma, pexels, archive)
        dest: Base destination directory
        extra_args: Additional arguments to pass

    Returns:
        True if successful
    """
    if source not in DATA_SOURCES:
        logger.error(f"Unknown source: {source}")
        return False

    config = DATA_SOURCES[source]
    script_path = get_script_path(config["script"])

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    # Build command
    args = [sys.executable, str(script_path)]

    # Add default args with destination substitution
    for arg in config["default_args"]:
        args.append(arg.format(dest=dest))

    # Add extra args
    if extra_args:
        args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Downloading from: {source}")
    print(f"Description: {config['description']}")
    print(f"Command: {' '.join(args)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(args, check=False)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run {source} download: {e}")
        return False


def download_all(
    dest: str,
    sources: List[str] = None,
    govdocs_range: str = None,
    fma_subset: str = None,
    pexels_count: int = None,
    archive_count: int = None
):
    """
    Download from all specified sources.

    Args:
        dest: Base destination directory
        sources: List of sources to download from (None = all)
        govdocs_range: Govdocs archive range
        fma_subset: FMA subset name
        pexels_count: Number of Pexels videos
        archive_count: Number of Internet Archive files
    """
    if sources is None:
        sources = list(DATA_SOURCES.keys())

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for source in sources:
        extra_args = []

        # Add source-specific arguments
        if source == "govdocs" and govdocs_range:
            extra_args = ["--range", govdocs_range]
        elif source == "fma" and fma_subset:
            extra_args = ["--subset", fma_subset]
        elif source == "pexels" and pexels_count:
            extra_args = ["--count", str(pexels_count)]
        elif source == "archive" and archive_count:
            extra_args = ["--count", str(archive_count)]

        success = run_download_script(source, str(dest_path), extra_args)
        results[source] = success

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for source, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {source}: {status}")

    return all(results.values())


def extract_features(
    input_dir: str,
    output_csv: str,
    file_types: List[str] = None,
    samples_per_file: int = 10,
    max_files: int = 0
):
    """
    Extract continuity features using Python extractor.

    Args:
        input_dir: Input directory with downloaded files
        output_csv: Output CSV path
        file_types: File types to process
        samples_per_file: Samples per file
        max_files: Maximum files to process
    """
    # Import feature extractor
    sys.path.insert(0, str(Path(__file__).parent.parent / "ml" / "continuity"))
    from feature_extractor import extract_dataset_from_directory

    print(f"\nExtracting features from {input_dir}")
    print(f"Output: {output_csv}")

    extract_dataset_from_directory(
        directory=input_dir,
        output_csv=output_csv,
        file_types=file_types,
        samples_per_file=samples_per_file,
        max_files=max_files
    )


def run_pipeline(
    dest: str,
    output_csv: str,
    sources: List[str] = None,
    use_cpp_extractor: bool = True,
    **kwargs
):
    """
    Run full data collection and feature extraction pipeline.

    Args:
        dest: Data destination directory
        output_csv: Output CSV path
        sources: Data sources to use
        use_cpp_extractor: Use C++ extractor (requires compiled binary)
        **kwargs: Additional arguments for download and extraction
    """
    print("=" * 60)
    print("Training Data Collection Pipeline")
    print("=" * 60)

    # Step 1: Download data
    print("\n[Step 1/2] Downloading training data...")
    if not download_all(dest, sources, **{k: v for k, v in kwargs.items()
                                          if k in ['govdocs_range', 'fma_subset',
                                                   'pexels_count', 'archive_count']}):
        print("Warning: Some downloads failed")

    # Step 2: Extract features
    print("\n[Step 2/2] Extracting features...")

    if use_cpp_extractor:
        # Try to use C++ extractor via CLI
        cli_path = Path(__file__).parent.parent / "Filerestore_CLI" / "x64" / "Release" / "Filerestore_CLI.exe"
        if cli_path.exists():
            print("Using C++ extractor (recommended)")
            cmd = [
                str(cli_path),
                "mlscan", dest,
                "--continuity",
                "--output", output_csv,
                "--samples-per-file", str(kwargs.get('samples_per_file', 10))
            ]
            subprocess.run(cmd)
        else:
            print("C++ extractor not found, falling back to Python")
            extract_features(
                input_dir=dest,
                output_csv=output_csv,
                samples_per_file=kwargs.get('samples_per_file', 10),
                max_files=kwargs.get('max_files', 0)
            )
    else:
        extract_features(
            input_dir=dest,
            output_csv=output_csv,
            samples_per_file=kwargs.get('samples_per_file', 10),
            max_files=kwargs.get('max_files', 0)
        )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Dataset saved to: {output_csv}")
    print("=" * 60)


def list_sources():
    """List available data sources."""
    print("Available Data Sources")
    print("=" * 60)
    for name, config in DATA_SOURCES.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  File types: {', '.join(config['file_types'])}")
        print(f"  Script: {config['script']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available sources
  python collect_training_data.py list

  # Download from all sources
  python collect_training_data.py download --dest ./data

  # Download specific sources
  python collect_training_data.py download --sources govdocs,fma --dest ./data

  # Extract features only
  python collect_training_data.py extract --input ./data --output dataset.csv

  # Full pipeline
  python collect_training_data.py pipeline --dest ./data --output dataset.csv

  # Quick test (small dataset)
  python collect_training_data.py pipeline --dest ./data --output test.csv \\
      --govdocs-range 001-002 --fma-subset small --pexels-count 50
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # List command
    subparsers.add_parser("list", help="List available data sources")

    # Download command
    dl = subparsers.add_parser("download", help="Download training data")
    dl.add_argument("--dest", "-d", required=True, help="Destination directory")
    dl.add_argument("--sources", "-s", help="Sources (comma-separated)")
    dl.add_argument("--govdocs-range", help="Govdocs range (e.g., 001-050)")
    dl.add_argument("--fma-subset", choices=["small", "medium", "large"],
                    help="FMA subset")
    dl.add_argument("--pexels-count", type=int, help="Number of Pexels videos")
    dl.add_argument("--archive-count", type=int, help="Number of Archive files")

    # Extract command
    ext = subparsers.add_parser("extract", help="Extract features from data")
    ext.add_argument("--input", "-i", required=True, help="Input directory")
    ext.add_argument("--output", "-o", required=True, help="Output CSV")
    ext.add_argument("--types", "-t", help="File types (comma-separated)")
    ext.add_argument("--samples", type=int, default=10, help="Samples per file")
    ext.add_argument("--max-files", type=int, default=0, help="Max files")

    # Pipeline command
    pipe = subparsers.add_parser("pipeline", help="Full download + extract pipeline")
    pipe.add_argument("--dest", "-d", required=True, help="Data directory")
    pipe.add_argument("--output", "-o", required=True, help="Output CSV")
    pipe.add_argument("--sources", "-s", help="Sources (comma-separated)")
    pipe.add_argument("--govdocs-range", default="001-010")
    pipe.add_argument("--fma-subset", default="small")
    pipe.add_argument("--pexels-count", type=int, default=500)
    pipe.add_argument("--archive-count", type=int, default=200)
    pipe.add_argument("--samples-per-file", type=int, default=10)
    pipe.add_argument("--use-python", action="store_true",
                      help="Force Python extractor instead of C++")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command == "list":
        list_sources()

    elif args.command == "download":
        sources = args.sources.split(',') if args.sources else None
        download_all(
            dest=args.dest,
            sources=sources,
            govdocs_range=args.govdocs_range,
            fma_subset=args.fma_subset,
            pexels_count=args.pexels_count,
            archive_count=args.archive_count
        )

    elif args.command == "extract":
        file_types = args.types.split(',') if args.types else None
        extract_features(
            input_dir=args.input,
            output_csv=args.output,
            file_types=file_types,
            samples_per_file=args.samples,
            max_files=args.max_files
        )

    elif args.command == "pipeline":
        sources = args.sources.split(',') if args.sources else None
        run_pipeline(
            dest=args.dest,
            output_csv=args.output,
            sources=sources,
            use_cpp_extractor=not args.use_python,
            govdocs_range=args.govdocs_range,
            fma_subset=args.fma_subset,
            pexels_count=args.pexels_count,
            archive_count=args.archive_count,
            samples_per_file=args.samples_per_file
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
