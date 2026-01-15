#!/usr/bin/env python3
"""
Unified Storage Interface for ML Training Workflow

Supports multiple storage backends:
- Local filesystem (本地存储)
- Aliyun OSS (阿里云对象存储)
- Tencent COS (腾讯云对象存储)
- Qiniu (七牛云)
- OneDrive (Microsoft Graph API)
- Google Drive (for Colab integration)

Usage:
    from storage import get_storage

    # Configure backend first
    # python storage.py config aliyun

    # Then use in code
    storage = get_storage("aliyun")
    storage.upload("local.csv", "ML/datasets/data.csv")
    storage.download("ML/models/model.onnx", "local.onnx")

CLI Usage:
    python storage.py config aliyun          # Configure backend
    python storage.py test aliyun            # Test connection
    python storage.py upload aliyun local.csv remote/path/
    python storage.py download aliyun remote/file.csv ./local/
    python storage.py list aliyun remote/path/
    python storage.py size aliyun            # Show total size
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CONFIG_DIR = Path.home() / ".filerestore"
CONFIG_FILE = CONFIG_DIR / "storage.json"


@dataclass
class FileInfo:
    """Information about a remote file."""
    key: str
    size: int
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None

    def __str__(self):
        return f"{self.key} ({format_size(self.size)})"


def format_size(size: int) -> str:
    """Format file size as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def load_config() -> Dict[str, Any]:
    """Load storage configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    """Save storage configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to {CONFIG_FILE}")


def list_backends() -> List[str]:
    """List configured backends."""
    config = load_config()
    return list(config.keys())


# =============================================================================
# Abstract Base Class
# =============================================================================

class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload(self, local_path: str, remote_path: str,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """Upload a file to remote storage."""
        pass

    @abstractmethod
    def download(self, remote_path: str, local_path: str,
                 progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """Download a file from remote storage."""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "", recursive: bool = True) -> Iterator[FileInfo]:
        """List files in remote storage."""
        pass

    @abstractmethod
    def delete(self, remote_path: str) -> bool:
        """Delete a file from remote storage."""
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists."""
        pass

    @abstractmethod
    def get_size(self, remote_path: str) -> int:
        """Get the size of a remote file."""
        pass

    def get_total_size(self, prefix: str = "") -> int:
        """Get total size of files with prefix."""
        return sum(f.size for f in self.list_files(prefix))

    def upload_directory(self, local_dir: str, remote_prefix: str,
                        pattern: str = "*",
                        progress_callback: Optional[Callable[[str, int, int], None]] = None) -> int:
        """Upload all files in a directory."""
        local_path = Path(local_dir)
        files = list(local_path.rglob(pattern))
        files = [f for f in files if f.is_file()]
        total = len(files)
        uploaded = 0

        for i, file in enumerate(files):
            rel_path = file.relative_to(local_path)
            remote_path = f"{remote_prefix.rstrip('/')}/{rel_path}".replace('\\', '/')

            if progress_callback:
                progress_callback(str(rel_path), i + 1, total)

            if self.upload(str(file), remote_path):
                uploaded += 1

        return uploaded

    def download_directory(self, remote_prefix: str, local_dir: str,
                          pattern: str = "*",
                          progress_callback: Optional[Callable[[str, int, int], None]] = None) -> int:
        """Download all files with a prefix."""
        import fnmatch

        files = list(self.list_files(remote_prefix))
        if pattern != "*":
            files = [f for f in files if fnmatch.fnmatch(f.key.split('/')[-1], pattern)]

        total = len(files)
        downloaded = 0
        local_path = Path(local_dir)

        for i, file_info in enumerate(files):
            rel_path = file_info.key
            if rel_path.startswith(remote_prefix):
                rel_path = rel_path[len(remote_prefix):].lstrip('/')

            local_file = local_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback(rel_path, i + 1, total)

            if self.download(file_info.key, str(local_file)):
                downloaded += 1

        return downloaded


# =============================================================================
# Local Storage Backend
# =============================================================================

class LocalBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, remote_path: str) -> Path:
        return self.base_path / remote_path.lstrip("/")

    def upload(self, local_path: str, remote_path: str,
               progress_callback: Optional[Callable] = None) -> bool:
        try:
            dest = self._resolve(remote_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: str,
                 progress_callback: Optional[Callable] = None) -> bool:
        try:
            src = self._resolve(remote_path)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, local_path)
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_files(self, prefix: str = "", recursive: bool = True) -> Iterator[FileInfo]:
        search_path = self._resolve(prefix) if prefix else self.base_path

        if search_path.is_file():
            stat = search_path.stat()
            yield FileInfo(
                key=prefix,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            )
            return

        if not search_path.exists():
            return

        pattern = "**/*" if recursive else "*"
        for path in search_path.glob(pattern):
            if path.is_file():
                rel_path = str(path.relative_to(self.base_path)).replace('\\', '/')
                stat = path.stat()
                yield FileInfo(
                    key=rel_path,
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime)
                )

    def delete(self, remote_path: str) -> bool:
        try:
            self._resolve(remote_path).unlink()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()

    def get_size(self, remote_path: str) -> int:
        try:
            return self._resolve(remote_path).stat().st_size
        except:
            return -1


# =============================================================================
# Aliyun OSS Backend (阿里云对象存储)
# =============================================================================

class AliyunOSSBackend(StorageBackend):
    """
    Aliyun OSS storage backend.

    Get credentials from: https://ram.console.aliyun.com/manage/ak
    Create bucket at: https://oss.console.aliyun.com/
    """

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        try:
            import oss2
            self.oss2 = oss2
        except ImportError:
            raise ImportError("Please install oss2: pip install oss2")

        auth = oss2.Auth(access_key, secret_key)
        self.bucket = oss2.Bucket(auth, endpoint, bucket)
        self.bucket_name = bucket

    def upload(self, local_path: str, remote_path: str,
               progress_callback: Optional[Callable] = None) -> bool:
        try:
            if progress_callback:
                self.bucket.put_object_from_file(
                    remote_path, local_path,
                    progress_callback=progress_callback
                )
            else:
                self.bucket.put_object_from_file(remote_path, local_path)
            logger.info(f"Uploaded {local_path} -> oss://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: str,
                 progress_callback: Optional[Callable] = None) -> bool:
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            if progress_callback:
                self.bucket.get_object_to_file(
                    remote_path, local_path,
                    progress_callback=progress_callback
                )
            else:
                self.bucket.get_object_to_file(remote_path, local_path)
            logger.info(f"Downloaded oss://{self.bucket_name}/{remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_files(self, prefix: str = "", recursive: bool = True) -> Iterator[FileInfo]:
        delimiter = '' if recursive else '/'
        for obj in self.oss2.ObjectIterator(self.bucket, prefix=prefix, delimiter=delimiter):
            if hasattr(obj, 'key') and not obj.key.endswith('/'):
                yield FileInfo(
                    key=obj.key,
                    size=obj.size,
                    last_modified=datetime.fromtimestamp(obj.last_modified) if obj.last_modified else None,
                    etag=obj.etag
                )

    def delete(self, remote_path: str) -> bool:
        try:
            self.bucket.delete_object(remote_path)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        return self.bucket.object_exists(remote_path)

    def get_size(self, remote_path: str) -> int:
        try:
            meta = self.bucket.head_object(remote_path)
            return meta.content_length
        except:
            return -1


# =============================================================================
# Tencent COS Backend (腾讯云对象存储)
# =============================================================================

class TencentCOSBackend(StorageBackend):
    """
    Tencent Cloud COS storage backend.

    Get credentials from: https://console.cloud.tencent.com/cam/capi
    Create bucket at: https://console.cloud.tencent.com/cos/bucket
    """

    def __init__(self, region: str, secret_id: str, secret_key: str, bucket: str):
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError:
            raise ImportError("Please install cos-python-sdk-v5: pip install cos-python-sdk-v5")

        config = CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
        )
        self.client = CosS3Client(config)
        self.bucket = bucket
        self.region = region

    def upload(self, local_path: str, remote_path: str,
               progress_callback: Optional[Callable] = None) -> bool:
        try:
            self.client.upload_file(
                Bucket=self.bucket,
                Key=remote_path,
                LocalFilePath=local_path
            )
            logger.info(f"Uploaded {local_path} -> cos://{self.bucket}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: str,
                 progress_callback: Optional[Callable] = None) -> bool:
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                Bucket=self.bucket,
                Key=remote_path,
                DestFilePath=local_path
            )
            logger.info(f"Downloaded cos://{self.bucket}/{remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_files(self, prefix: str = "", recursive: bool = True) -> Iterator[FileInfo]:
        marker = ""
        while True:
            response = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=prefix,
                Marker=marker,
                Delimiter='' if recursive else '/'
            )

            for obj in response.get('Contents', []):
                if not obj['Key'].endswith('/'):
                    yield FileInfo(
                        key=obj['Key'],
                        size=int(obj['Size']),
                        etag=obj.get('ETag', '').strip('"')
                    )

            if response.get('IsTruncated') == 'false':
                break
            marker = response.get('NextMarker', '')
            if not marker:
                break

    def delete(self, remote_path: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except:
            return False

    def get_size(self, remote_path: str) -> int:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return int(response.get('Content-Length', -1))
        except:
            return -1


# =============================================================================
# Qiniu Backend (七牛云)
# =============================================================================

class QiniuBackend(StorageBackend):
    """
    Qiniu cloud storage backend.

    Get credentials from: https://portal.qiniu.com/user/key
    Create bucket at: https://portal.qiniu.com/kodo/bucket
    """

    def __init__(self, access_key: str, secret_key: str, bucket: str, domain: str):
        try:
            import qiniu
            self.qiniu = qiniu
        except ImportError:
            raise ImportError("Please install qiniu: pip install qiniu")

        self.auth = qiniu.Auth(access_key, secret_key)
        self.bucket_name = bucket
        self.domain = domain
        self.bucket_manager = qiniu.BucketManager(self.auth)

    def upload(self, local_path: str, remote_path: str,
               progress_callback: Optional[Callable] = None) -> bool:
        try:
            token = self.auth.upload_token(self.bucket_name, remote_path)
            ret, info = self.qiniu.put_file(token, remote_path, local_path)
            if ret and 'key' in ret:
                logger.info(f"Uploaded {local_path} -> qiniu://{self.bucket_name}/{remote_path}")
                return True
            else:
                logger.error(f"Upload failed: {info}")
                return False
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download(self, remote_path: str, local_path: str,
                 progress_callback: Optional[Callable] = None) -> bool:
        try:
            import requests
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            base_url = f"http://{self.domain}/{remote_path}"
            private_url = self.auth.private_download_url(base_url, expires=3600)

            response = requests.get(private_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)

            logger.info(f"Downloaded qiniu://{self.bucket_name}/{remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_files(self, prefix: str = "", recursive: bool = True) -> Iterator[FileInfo]:
        marker = None
        while True:
            ret, eof, info = self.bucket_manager.list(
                self.bucket_name,
                prefix=prefix,
                marker=marker,
                limit=1000,
                delimiter=None if recursive else '/'
            )

            if ret is None:
                break

            for item in ret.get('items', []):
                yield FileInfo(
                    key=item['key'],
                    size=item['fsize'],
                    last_modified=datetime.fromtimestamp(item['putTime'] / 10000000) if 'putTime' in item else None,
                    etag=item.get('hash')
                )

            if eof:
                break
            marker = ret.get('marker')

    def delete(self, remote_path: str) -> bool:
        try:
            ret, info = self.bucket_manager.delete(self.bucket_name, remote_path)
            return info.status_code == 200
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def exists(self, remote_path: str) -> bool:
        ret, info = self.bucket_manager.stat(self.bucket_name, remote_path)
        return ret is not None

    def get_size(self, remote_path: str) -> int:
        ret, info = self.bucket_manager.stat(self.bucket_name, remote_path)
        if ret:
            return ret.get('fsize', -1)
        return -1


# =============================================================================
# Factory Function
# =============================================================================

BACKEND_CLASSES = {
    'local': LocalBackend,
    'aliyun': AliyunOSSBackend,
    'oss': AliyunOSSBackend,
    'tencent': TencentCOSBackend,
    'cos': TencentCOSBackend,
    'qiniu': QiniuBackend,
}


def get_storage(backend: str, **kwargs) -> StorageBackend:
    """
    Get a storage backend instance.

    Args:
        backend: Backend name ('local', 'aliyun', 'tencent', 'qiniu')
        **kwargs: Override config values

    Returns:
        StorageBackend instance
    """
    backend = backend.lower()

    # Handle aliases
    if backend == 'oss':
        backend = 'aliyun'
    if backend == 'cos':
        backend = 'tencent'

    if backend == 'local':
        base_path = kwargs.get('base_path', './storage')
        return LocalBackend(base_path=base_path)

    # Load config
    config = load_config()
    if backend not in config:
        raise ValueError(f"Backend '{backend}' not configured. Run: python storage.py config {backend}")

    cfg = config[backend]

    # Merge with kwargs
    cfg.update(kwargs)

    if backend == 'aliyun':
        return AliyunOSSBackend(
            endpoint=cfg['endpoint'],
            access_key=cfg['access_key'],
            secret_key=cfg['secret_key'],
            bucket=cfg['bucket']
        )
    elif backend == 'tencent':
        return TencentCOSBackend(
            region=cfg['region'],
            secret_id=cfg['access_key'],
            secret_key=cfg['secret_key'],
            bucket=cfg['bucket']
        )
    elif backend == 'qiniu':
        return QiniuBackend(
            access_key=cfg['access_key'],
            secret_key=cfg['secret_key'],
            bucket=cfg['bucket'],
            domain=cfg['domain']
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# CLI
# =============================================================================

# Template config for reference
CONFIG_TEMPLATE = {
    "tencent": {
        "region": "ap-guangzhou",
        "access_key": "YOUR_SECRET_ID",
        "secret_key": "YOUR_SECRET_KEY",
        "bucket": "your-bucket-name-1234567890"
    },
    "aliyun": {
        "endpoint": "oss-cn-hangzhou.aliyuncs.com",
        "access_key": "YOUR_ACCESS_KEY_ID",
        "secret_key": "YOUR_ACCESS_KEY_SECRET",
        "bucket": "your-bucket-name"
    },
    "qiniu": {
        "access_key": "YOUR_ACCESS_KEY",
        "secret_key": "YOUR_SECRET_KEY",
        "bucket": "your-bucket-name",
        "domain": "your-domain.bkt.clouddn.com"
    }
}


def cmd_config_import(args):
    """Import configuration from a JSON file."""
    import_path = Path(args.file)

    if not import_path.exists():
        print(f"Error: File not found: {import_path}")
        sys.exit(1)

    try:
        with open(import_path, 'r', encoding='utf-8') as f:
            new_config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    # Validate the config structure
    valid_backends = ['tencent', 'aliyun', 'qiniu', 'local']
    imported = []

    for backend, cfg in new_config.items():
        if backend not in valid_backends:
            print(f"Warning: Unknown backend '{backend}', skipping")
            continue

        # Check required fields
        if backend == 'tencent':
            required = ['region', 'access_key', 'secret_key', 'bucket']
        elif backend == 'aliyun':
            required = ['endpoint', 'access_key', 'secret_key', 'bucket']
        elif backend == 'qiniu':
            required = ['access_key', 'secret_key', 'bucket', 'domain']
        else:
            required = []

        missing = [f for f in required if f not in cfg]
        if missing:
            print(f"Warning: Backend '{backend}' missing fields: {missing}, skipping")
            continue

        imported.append(backend)

    if not imported:
        print("Error: No valid backend configurations found")
        sys.exit(1)

    # Merge with existing config (if --merge flag) or replace
    if args.merge:
        config = load_config()
        for backend in imported:
            config[backend] = new_config[backend]
    else:
        config = {k: v for k, v in new_config.items() if k in imported}

    save_config(config)
    print(f"\nImported configurations: {', '.join(imported)}")
    print(f"Config saved to: {CONFIG_FILE}")


def cmd_config_export(args):
    """Export current configuration to a JSON file."""
    config = load_config()

    if not config:
        print("No configuration to export")
        return

    export_path = Path(args.file)

    # Optionally mask secrets
    if args.mask_secrets:
        config = json.loads(json.dumps(config))  # Deep copy
        for backend, cfg in config.items():
            if 'secret_key' in cfg:
                cfg['secret_key'] = cfg['secret_key'][:4] + '****' + cfg['secret_key'][-4:] if len(cfg['secret_key']) > 8 else '****'
            if 'access_key' in cfg:
                cfg['access_key'] = cfg['access_key'][:4] + '****' + cfg['access_key'][-4:] if len(cfg['access_key']) > 8 else '****'

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Configuration exported to: {export_path}")
    if args.mask_secrets:
        print("(Secrets have been masked)")


def cmd_config_show(args):
    """Show current configuration."""
    print(f"\nConfig file: {CONFIG_FILE}")
    print(f"Config directory: {CONFIG_DIR}")

    config = load_config()

    if not config:
        print("\nNo configuration found.")
        print(f"\nTo configure, either:")
        print(f"  1. Run: python storage.py config <backend>")
        print(f"  2. Import from JSON: python storage.py config-import config.json")
        print(f"  3. Create template: python storage.py config-template")
        return

    print(f"\nConfigured backends: {list(config.keys())}")

    # Show config with masked secrets
    for backend, cfg in config.items():
        print(f"\n[{backend}]")
        for key, value in cfg.items():
            if 'secret' in key.lower() or 'key' in key.lower():
                # Mask sensitive values
                if len(str(value)) > 8:
                    masked = str(value)[:4] + '****' + str(value)[-4:]
                else:
                    masked = '****'
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")


def cmd_config_template(args):
    """Generate a template configuration file."""
    template_path = Path(args.file) if args.file else Path("storage_config_template.json")

    # Only include requested backends or all
    if args.backend:
        backends = [b.strip() for b in args.backend.split(',')]
        template = {k: v for k, v in CONFIG_TEMPLATE.items() if k in backends}
    else:
        template = CONFIG_TEMPLATE

    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Template saved to: {template_path}")
    print("\nEdit this file with your credentials, then import with:")
    print(f"  python storage.py config-import {template_path}")


def cmd_config(args):
    """Interactive configuration."""
    backend = args.backend.lower()
    config = load_config()

    print(f"\n=== Configure {backend} ===\n")

    if backend in ('aliyun', 'oss'):
        print("Get credentials from: https://ram.console.aliyun.com/manage/ak")
        print("Create bucket at: https://oss.console.aliyun.com/\n")

        access_key = input("Access Key ID: ").strip()
        secret_key = input("Access Key Secret: ").strip()
        endpoint = input("Endpoint (e.g., oss-cn-hangzhou.aliyuncs.com): ").strip()
        bucket = input("Bucket name: ").strip()

        config['aliyun'] = {
            'endpoint': endpoint,
            'access_key': access_key,
            'secret_key': secret_key,
            'bucket': bucket
        }

    elif backend in ('tencent', 'cos'):
        print("Get credentials from: https://console.cloud.tencent.com/cam/capi")
        print("Create bucket at: https://console.cloud.tencent.com/cos/bucket\n")

        secret_id = input("SecretId: ").strip()
        secret_key = input("SecretKey: ").strip()
        region = input("Region (e.g., ap-guangzhou): ").strip()
        bucket = input("Bucket name (with appid, e.g., mybucket-1234567890): ").strip()

        config['tencent'] = {
            'region': region,
            'access_key': secret_id,
            'secret_key': secret_key,
            'bucket': bucket
        }

    elif backend == 'qiniu':
        print("Get credentials from: https://portal.qiniu.com/user/key")
        print("Create bucket at: https://portal.qiniu.com/kodo/bucket\n")

        access_key = input("Access Key: ").strip()
        secret_key = input("Secret Key: ").strip()
        bucket = input("Bucket name: ").strip()
        domain = input("Download domain (e.g., xxx.bkt.clouddn.com): ").strip()

        config['qiniu'] = {
            'access_key': access_key,
            'secret_key': secret_key,
            'bucket': bucket,
            'domain': domain
        }

    elif backend == 'local':
        base_path = input("Base path (default: ./storage): ").strip() or "./storage"
        config['local'] = {'base_path': base_path}

    else:
        print(f"Unknown backend: {backend}")
        print(f"Available: {list(BACKEND_CLASSES.keys())}")
        return

    save_config(config)
    print(f"\n{backend} configured successfully!")

    test = input("\nTest connection now? (y/n): ").strip().lower()
    if test == 'y':
        cmd_test(args)


def cmd_test(args):
    """Test connection to backend."""
    try:
        storage = get_storage(args.backend)

        print(f"\nTesting {args.backend}...")

        # List files (basic connectivity test)
        files = list(storage.list_files("", recursive=False))
        print(f"Connection successful!")
        print(f"Found {len(files)} items in root")

        # Show total size
        total_size = storage.get_total_size("")
        print(f"Total storage used: {format_size(total_size)}")

    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_list(args):
    """List files in remote storage."""
    storage = get_storage(args.backend)

    prefix = args.path or ""
    total_size = 0
    count = 0

    print(f"\nFiles in {args.backend}:{prefix}\n")

    for f in storage.list_files(prefix):
        print(f"  {f}")
        total_size += f.size
        count += 1

    print(f"\nTotal: {count} files, {format_size(total_size)}")


def cmd_upload(args):
    """Upload file or directory."""
    storage = get_storage(args.backend)

    local_path = Path(args.local)
    remote_path = args.remote

    if local_path.is_dir():
        print(f"Uploading directory {local_path} to {args.backend}:{remote_path}")

        def progress(filename, num, total):
            print(f"  [{num}/{total}] {filename}")

        count = storage.upload_directory(str(local_path), remote_path, progress_callback=progress)
        print(f"\nUploaded {count} files")
    else:
        print(f"Uploading {local_path} to {args.backend}:{remote_path}")

        # If remote_path ends with /, append filename
        if remote_path.endswith('/'):
            remote_path = remote_path + local_path.name

        if storage.upload(str(local_path), remote_path):
            print("Upload successful!")
        else:
            print("Upload failed!")
            sys.exit(1)


def cmd_download(args):
    """Download file or directory."""
    storage = get_storage(args.backend)

    remote_path = args.remote
    local_path = args.local

    # Check if it's a directory (multiple files)
    files = list(storage.list_files(remote_path))

    if len(files) > 1 or (len(files) == 1 and files[0].key != remote_path):
        print(f"Downloading {args.backend}:{remote_path} to {local_path}")

        def progress(filename, num, total):
            print(f"  [{num}/{total}] {filename}")

        count = storage.download_directory(remote_path, local_path, progress_callback=progress)
        print(f"\nDownloaded {count} files")
    else:
        print(f"Downloading {args.backend}:{remote_path} to {local_path}")

        if storage.download(remote_path, local_path):
            print("Download successful!")
        else:
            print("Download failed!")
            sys.exit(1)


def cmd_delete(args):
    """Delete a file."""
    storage = get_storage(args.backend)

    if storage.delete(args.remote):
        print(f"Deleted {args.remote}")
    else:
        print("Delete failed!")
        sys.exit(1)


def cmd_delete_prefix(args):
    """Delete all files with a given prefix."""
    storage = get_storage(args.backend)

    prefix = args.prefix
    files = list(storage.list_files(prefix))

    if not files:
        print(f"No files found with prefix: {prefix}")
        return

    print(f"\nFiles to delete ({len(files)} total):")
    total_size = 0
    for f in files[:10]:  # Show first 10
        print(f"  {f}")
        total_size += f.size
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
        total_size = sum(f.size for f in files)

    print(f"\nTotal size: {format_size(total_size)}")

    if not args.yes:
        confirm = input("\nAre you sure you want to delete these files? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            return

    print("\nDeleting...")
    deleted = 0
    failed = 0
    for f in files:
        if storage.delete(f.key):
            deleted += 1
            print(f"  Deleted: {f.key}")
        else:
            failed += 1
            print(f"  FAILED: {f.key}")

    print(f"\nDone. Deleted: {deleted}, Failed: {failed}")


def cmd_size(args):
    """Show total size."""
    storage = get_storage(args.backend)

    prefix = args.path or ""
    total_size = storage.get_total_size(prefix)

    print(f"Total size of {args.backend}:{prefix}: {format_size(total_size)}")


def cmd_backends(args):
    """List configured backends."""
    backends = list_backends()

    if not backends:
        print("No backends configured.")
        print("Run: python storage.py config <backend>")
        print(f"Available backends: {list(BACKEND_CLASSES.keys())}")
        return

    print("Configured backends:")
    for name in backends:
        print(f"  - {name}")


def cmd_batch_transfer(args):
    """
    Batch download from source, upload to cloud, delete local, repeat.
    Useful for transferring large datasets with limited local disk space.
    """
    storage = get_storage(args.backend)
    temp_dir = Path(args.temp_dir)
    batch_size_bytes = args.batch_size * 1024 * 1024 * 1024  # Convert GB to bytes

    # Parse source URLs
    sources = []
    if args.source_script:
        # Use download script
        sources = [{'type': 'script', 'script': args.source_script, 'args': args.source_args or []}]
    elif args.source_urls:
        # Direct URLs
        sources = [{'type': 'url', 'url': url} for url in args.source_urls.split(',')]

    print(f"\n{'='*60}")
    print(f"Batch Transfer to {args.backend}")
    print(f"{'='*60}")
    print(f"Temp directory: {temp_dir}")
    print(f"Batch size: {args.batch_size} GB")
    print(f"Remote prefix: {args.remote_prefix}")
    print(f"{'='*60}\n")

    total_uploaded = 0
    batch_num = 0

    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for source in sources:
            if source['type'] == 'script':
                # Run download script
                script = source['script']
                script_args = source['args']

                print(f"\n[Batch {batch_num + 1}] Running: {script} {' '.join(script_args)}")

                # Run the download script
                cmd = [sys.executable, script] + script_args + ['--dest', str(temp_dir)]
                result = subprocess.run(cmd, capture_output=False)

                if result.returncode != 0:
                    print(f"Warning: Script exited with code {result.returncode}")

            # Check temp directory size
            current_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
            print(f"Downloaded: {format_size(current_size)}")

            # Upload when batch is ready or this is the last source
            if current_size >= batch_size_bytes or source == sources[-1]:
                batch_num += 1
                print(f"\n[Batch {batch_num}] Uploading {format_size(current_size)} to {args.backend}...")

                def progress(filename, num, total):
                    print(f"  [{num}/{total}] {filename}")

                count = storage.upload_directory(
                    str(temp_dir),
                    args.remote_prefix,
                    progress_callback=progress
                )

                total_uploaded += count
                print(f"Uploaded {count} files")

                # Clean temp directory
                print(f"Cleaning temp directory...")
                for f in temp_dir.rglob('*'):
                    if f.is_file():
                        f.unlink()
                for d in sorted(temp_dir.rglob('*'), reverse=True):
                    if d.is_dir():
                        try:
                            d.rmdir()
                        except:
                            pass

                print(f"Batch {batch_num} complete!\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Cleaning up...")

    finally:
        # Final cleanup
        if temp_dir.exists():
            for f in temp_dir.rglob('*'):
                if f.is_file():
                    f.unlink()

    print(f"\n{'='*60}")
    print(f"Transfer Complete!")
    print(f"Total batches: {batch_num}")
    print(f"Total files uploaded: {total_uploaded}")
    print(f"{'='*60}")


def cmd_batch_download_upload(args):
    """
    Download data sources in batches, upload to cloud, delete local.
    Designed for govdocs/fma/pexels style downloads.
    """
    storage = get_storage(args.backend)
    temp_dir = Path(args.temp_dir)
    batch_size_gb = args.batch_size

    print(f"\n{'='*60}")
    print(f"Batch Download & Upload")
    print(f"{'='*60}")
    print(f"Backend: {args.backend}")
    print(f"Data source: {args.data_source}")
    print(f"Batch size: {batch_size_gb} GB")
    print(f"Temp directory: {temp_dir}")
    print(f"Remote prefix: {args.remote_prefix}")
    print(f"{'='*60}\n")

    temp_dir.mkdir(parents=True, exist_ok=True)

    total_uploaded = 0
    batch_num = 0

    if args.data_source == 'govdocs':
        # Download govdocs in ranges
        start = args.start or 1
        end = args.end or 1000
        batch_range = args.batch_range or 5  # 5 archives per batch (~2.5GB)

        current = start
        while current <= end:
            batch_end = min(current + batch_range - 1, end)
            batch_num += 1

            print(f"\n[Batch {batch_num}] Downloading govdocs {current:03d}-{batch_end:03d}...")

            # Download
            cmd = [
                sys.executable, 'scripts/download_govdocs.py', 'download',
                '--range', f'{current:03d}-{batch_end:03d}',
                '--dest', str(temp_dir / 'govdocs')
            ]
            result = subprocess.run(cmd)

            # Check if any files were downloaded
            govdocs_dir = temp_dir / 'govdocs'
            if not govdocs_dir.exists() or not any(govdocs_dir.iterdir()):
                print(f"\nWARNING: No files downloaded for batch {batch_num}, skipping upload.")
                current = batch_end + 1
                continue

            # Count downloaded files
            downloaded_files = list(govdocs_dir.glob('*.zip'))
            expected_count = batch_end - current + 1
            if len(downloaded_files) < expected_count:
                print(f"\nWARNING: Only {len(downloaded_files)}/{expected_count} files downloaded.")
                print("Uploading available files anyway...")

            # Upload
            print(f"Uploading to {args.backend}:{args.remote_prefix}...")
            def progress(filename, num, total):
                print(f"  [{num}/{total}] {filename}")

            count = storage.upload_directory(
                str(temp_dir),
                args.remote_prefix,
                progress_callback=progress
            )
            total_uploaded += count

            # Clean
            print("Cleaning temp directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            current = batch_end + 1

    elif args.data_source == 'fma':
        # FMA is typically downloaded as a whole subset
        batch_num = 1
        subset = args.subset or 'small'

        # Expected sizes for validation (approximate)
        expected_sizes = {
            'small': 7.2 * 1024 * 1024 * 1024,   # 7.2 GB
            'medium': 22 * 1024 * 1024 * 1024,   # 22 GB
            'large': 93 * 1024 * 1024 * 1024,    # 93 GB
        }

        print(f"\n[Batch 1] Downloading FMA {subset}...")
        cmd = [
            sys.executable, 'scripts/download_fma.py', 'download',
            '--subset', subset,
            '--dest', str(temp_dir / 'fma')
        ]
        result = subprocess.run(cmd)

        # Check if download succeeded
        if result.returncode != 0:
            print(f"\nERROR: FMA download failed (exit code {result.returncode})")
            print("Skipping upload of incomplete data.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return

        # Validate downloaded file size
        zip_file = temp_dir / 'fma' / f'fma_{subset}.zip'
        if zip_file.exists():
            actual_size = zip_file.stat().st_size
            expected = expected_sizes.get(subset, 0)
            if expected > 0 and actual_size < expected * 0.95:  # Allow 5% tolerance
                print(f"\nERROR: Downloaded file appears incomplete!")
                print(f"  Expected: ~{format_size(expected)}")
                print(f"  Actual:   {format_size(actual_size)}")
                print("Skipping upload. Please retry the download.")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return

        # Upload
        print(f"Uploading to {args.backend}:{args.remote_prefix}...")
        def progress(filename, num, total):
            print(f"  [{num}/{total}] {filename}")

        count = storage.upload_directory(
            str(temp_dir),
            args.remote_prefix,
            progress_callback=progress
        )
        total_uploaded += count

        # Clean
        shutil.rmtree(temp_dir, ignore_errors=True)

    elif args.data_source == 'pexels':
        # Pexels in batches
        total_count = args.count or 1000
        batch_count = args.batch_count or 100

        current = 0
        while current < total_count:
            batch_num += 1
            this_batch = min(batch_count, total_count - current)

            print(f"\n[Batch {batch_num}] Downloading {this_batch} Pexels videos...")
            cmd = [
                sys.executable, 'scripts/download_pexels.py', 'download',
                '--count', str(this_batch),
                '--dest', str(temp_dir / 'pexels')
            ]
            subprocess.run(cmd)

            # Upload
            print(f"Uploading to {args.backend}:{args.remote_prefix}...")
            def progress(filename, num, total):
                print(f"  [{num}/{total}] {filename}")

            count = storage.upload_directory(
                str(temp_dir),
                args.remote_prefix,
                progress_callback=progress
            )
            total_uploaded += count

            # Clean
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)

            current += this_batch

    elif args.data_source == 'archive':
        # Internet Archive
        batch_num = 1
        collection = args.collection or 'audio'
        count = args.count or 500

        print(f"\n[Batch 1] Downloading {count} files from Archive ({collection})...")
        cmd = [
            sys.executable, 'scripts/download_archive.py', 'download',
            '--collection', collection,
            '--count', str(count),
            '--dest', str(temp_dir / 'archive')
        ]
        subprocess.run(cmd)

        # Upload
        print(f"Uploading to {args.backend}:{args.remote_prefix}...")
        def progress(filename, num, total):
            print(f"  [{num}/{total}] {filename}")

        count = storage.upload_directory(
            str(temp_dir),
            args.remote_prefix,
            progress_callback=progress
        )
        total_uploaded += count

        # Clean
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Batch Transfer Complete!")
    print(f"Total batches: {batch_num}")
    print(f"Total files uploaded: {total_uploaded}")

    # Show final size
    total_size = storage.get_total_size(args.remote_prefix)
    print(f"Total size on {args.backend}: {format_size(total_size)}")
    print(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Cloud Storage CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Configure backends (interactive)
  python storage.py config tencent          # Interactive configuration

  # Configure backends (from JSON file) - RECOMMENDED
  python storage.py config-template -o my_config.json   # Create template
  # Edit my_config.json with your credentials
  python storage.py config-import my_config.json        # Import config

  # View/Export config
  python storage.py config-show                         # Show current config
  python storage.py config-export backup.json           # Export to file

  # Test connection
  python storage.py test tencent

  # List configured backends
  python storage.py backends

  # File operations
  python storage.py list tencent ML/
  python storage.py upload tencent data.csv ML/datasets/data.csv
  python storage.py upload tencent ./data/ ML/data/          # Upload directory
  python storage.py download tencent ML/models/ ./models/    # Download directory
  python storage.py size tencent ML/

  # Batch transfer (download -> upload -> delete -> repeat)
  python storage.py batch tencent govdocs --start 1 --end 100 --batch-range 10
  python storage.py batch tencent fma --subset medium
  python storage.py batch tencent pexels --count 1000 --batch-count 100

Config file location: ~/.filerestore/storage.json
"""
    )

    subparsers = parser.add_subparsers(dest="command")

    # config (interactive)
    p = subparsers.add_parser("config", help="Configure a backend interactively")
    p.add_argument("backend", help="Backend name (aliyun, tencent, qiniu, local)")
    p.set_defaults(func=cmd_config)

    # config-import
    p = subparsers.add_parser("config-import", help="Import config from JSON file")
    p.add_argument("file", help="JSON config file to import")
    p.add_argument("--merge", "-m", action="store_true",
                   help="Merge with existing config instead of replacing")
    p.set_defaults(func=cmd_config_import)

    # config-export
    p = subparsers.add_parser("config-export", help="Export config to JSON file")
    p.add_argument("file", help="Output JSON file")
    p.add_argument("--mask-secrets", "-m", action="store_true",
                   help="Mask sensitive values in export")
    p.set_defaults(func=cmd_config_export)

    # config-show
    p = subparsers.add_parser("config-show", help="Show current configuration")
    p.set_defaults(func=cmd_config_show)

    # config-template
    p = subparsers.add_parser("config-template", help="Generate a template config file")
    p.add_argument("-o", "--file", help="Output file (default: storage_config_template.json)")
    p.add_argument("-b", "--backend", help="Specific backends (comma-separated)")
    p.set_defaults(func=cmd_config_template)

    # test
    p = subparsers.add_parser("test", help="Test backend connection")
    p.add_argument("backend", help="Backend name")
    p.set_defaults(func=cmd_test)

    # backends
    p = subparsers.add_parser("backends", help="List configured backends")
    p.set_defaults(func=cmd_backends)

    # list
    p = subparsers.add_parser("list", help="List files")
    p.add_argument("backend", help="Backend name")
    p.add_argument("path", nargs="?", default="", help="Path prefix")
    p.set_defaults(func=cmd_list)

    # upload
    p = subparsers.add_parser("upload", help="Upload file or directory")
    p.add_argument("backend", help="Backend name")
    p.add_argument("local", help="Local path")
    p.add_argument("remote", help="Remote path")
    p.set_defaults(func=cmd_upload)

    # download
    p = subparsers.add_parser("download", help="Download file or directory")
    p.add_argument("backend", help="Backend name")
    p.add_argument("remote", help="Remote path")
    p.add_argument("local", help="Local path")
    p.set_defaults(func=cmd_download)

    # delete
    p = subparsers.add_parser("delete", help="Delete a file")
    p.add_argument("backend", help="Backend name")
    p.add_argument("remote", help="Remote path")
    p.set_defaults(func=cmd_delete)

    # delete-prefix
    p = subparsers.add_parser("delete-prefix", help="Delete all files with a prefix")
    p.add_argument("backend", help="Backend name")
    p.add_argument("prefix", help="Remote path prefix to delete")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_delete_prefix)

    # size
    p = subparsers.add_parser("size", help="Show total size")
    p.add_argument("backend", help="Backend name")
    p.add_argument("path", nargs="?", default="", help="Path prefix")
    p.set_defaults(func=cmd_size)

    # batch - batch download and upload
    p = subparsers.add_parser("batch", help="Batch download -> upload -> delete -> repeat")
    p.add_argument("backend", help="Backend name")
    p.add_argument("data_source", choices=['govdocs', 'fma', 'pexels', 'archive'],
                   help="Data source to download")
    p.add_argument("--temp-dir", "-t", default="D:\\temp\\batch_transfer",
                   help="Temporary directory for downloads (default: D:\\temp\\batch_transfer)")
    p.add_argument("--remote-prefix", "-r", default="ML/",
                   help="Remote path prefix (default: ML/)")
    p.add_argument("--batch-size", "-b", type=int, default=5,
                   help="Batch size in GB (default: 5)")
    # Govdocs options
    p.add_argument("--start", type=int, help="Govdocs: start archive number")
    p.add_argument("--end", type=int, help="Govdocs: end archive number")
    p.add_argument("--batch-range", type=int, default=10,
                   help="Govdocs: archives per batch (default: 10, ~5GB)")
    # FMA options
    p.add_argument("--subset", choices=['small', 'medium', 'large'],
                   help="FMA: subset to download")
    # Pexels options
    p.add_argument("--count", type=int, help="Pexels/Archive: total files to download")
    p.add_argument("--batch-count", type=int, default=100,
                   help="Pexels: files per batch (default: 100)")
    # Archive options
    p.add_argument("--collection", choices=['audio', 'movies'],
                   help="Archive: collection to download")
    p.set_defaults(func=cmd_batch_download_upload)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
