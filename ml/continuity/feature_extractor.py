#!/usr/bin/env python3
"""
Block Continuity Feature Extractor (Python)

Python implementation of 64-dimensional continuity feature extraction,
compatible with the C++ BlockContinuityDetector.

Features:
- Block 1 features (0-15): Statistics of first block
- Block 2 features (16-31): Statistics of second block
- Boundary features (32-47): Cross-boundary analysis
- Format-specific features (48-63): ZIP/MP3/MP4 specific

Usage:
    from feature_extractor import ContinuityFeatureExtractor

    extractor = ContinuityFeatureExtractor()
    features = extractor.extract(block1_data, block2_data, file_type="mp3")

    # Or from file
    samples = extractor.extract_from_file("test.mp3", samples_per_file=10)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import struct


class ContinuityFeatureExtractor:
    """
    Extract 64-dimensional continuity features from block pairs.

    Compatible with C++ BlockContinuityDetector.
    """

    FEATURE_DIM = 64
    DEFAULT_BLOCK_SIZE = 8192  # 8KB

    # ZIP-based formats
    ZIP_FORMATS = {'zip', 'docx', 'xlsx', 'pptx', 'jar', 'apk'}

    # Audio formats
    AUDIO_FORMATS = {'mp3', 'mp2', 'mp1', 'wav', 'flac', 'ogg', 'm4a'}

    # Video formats
    VIDEO_FORMATS = {'mp4', 'mov', 'avi', 'mkv', 'webm', '3gp', 'm4v'}

    # Image formats
    IMAGE_FORMATS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        self.block_size = block_size

    def extract(
        self,
        block1: bytes,
        block2: bytes,
        file_type: str = "unknown"
    ) -> np.ndarray:
        """
        Extract 64-dimensional features from two adjacent blocks.

        Args:
            block1: First block (tail of previous chunk)
            block2: Second block (head of next chunk)
            file_type: File extension (e.g., "mp3", "zip")

        Returns:
            64-dimensional feature vector
        """
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Convert to numpy arrays
        b1 = np.frombuffer(block1, dtype=np.uint8)
        b2 = np.frombuffer(block2, dtype=np.uint8)

        # Block 1 features (0-15)
        self._extract_block_features(b1, features, 0, file_type)

        # Block 2 features (16-31)
        self._extract_block_features(b2, features, 16, file_type)

        # Boundary features (32-47)
        self._extract_boundary_features(b1, b2, features, 32, file_type)

        # Format-specific features (48-63)
        file_type_lower = file_type.lower()
        if file_type_lower in self.ZIP_FORMATS:
            self._extract_zip_features(b1, b2, features, 48)
        elif file_type_lower in {'mp3', 'mp2', 'mp1'}:
            self._extract_mp3_features(b1, b2, features, 48)
        elif file_type_lower in {'mp4', 'mov', 'm4a', 'm4v', '3gp'}:
            self._extract_mp4_features(b1, b2, features, 48)
        else:
            self._extract_generic_features(b1, b2, features, 48)

        return features

    def _extract_block_features(
        self,
        data: np.ndarray,
        features: np.ndarray,
        offset: int,
        file_type: str
    ):
        """Extract 16 features from a single block (matches C++ implementation)."""
        if len(data) == 0:
            return

        # Entropy (normalized to [0, 1])
        features[offset + 0] = self._calculate_entropy(data)

        # Mean [0-1]
        features[offset + 1] = data.mean() / 255.0

        # Std dev [0-1] (normalized by 128, same as C++)
        features[offset + 2] = data.std() / 128.0

        # Zero byte ratio
        features[offset + 3] = np.sum(data == 0) / len(data)

        # High byte ratio (>= 0x80)
        features[offset + 4] = np.sum(data >= 128) / len(data)

        # Printable ratio (32-126)
        printable = np.sum((data >= 32) & (data <= 126))
        features[offset + 5] = printable / len(data)

        # 8-bin histogram (6-13) - matches C++ implementation
        # C++ uses: bin = data[i] / 32 (256 / 8 = 32)
        bins = np.zeros(8, dtype=np.float32)
        for b in data:
            bin_idx = min(b // 32, 7)
            bins[bin_idx] += 1
        bins /= len(data)
        features[offset + 6:offset + 14] = bins

        # PK signature detection (matches C++ - checks valid ZIP signatures)
        features[offset + 14] = self._detect_pk_signature_strict(data)

        # Compression score (matches C++ logic)
        features[offset + 15] = self._detect_compression_score(data)

    def _extract_boundary_features(
        self,
        block1: np.ndarray,
        block2: np.ndarray,
        features: np.ndarray,
        offset: int,
        file_type: str
    ):
        """Extract 16 boundary features (matches C++ implementation)."""
        if len(block1) == 0 or len(block2) == 0:
            return

        # Entropy values
        e1 = self._calculate_entropy(block1)
        e2 = self._calculate_entropy(block2)

        # Entropy difference
        features[offset + 0] = abs(e1 - e2)

        # Entropy gradient (normalized to [0,1], matches C++)
        features[offset + 1] = (e2 - e1 + 1.0) / 2.0

        # Mean difference (normalized by 255, matches C++)
        m1 = block1.mean()
        m2 = block2.mean()
        features[offset + 2] = abs(m1 - m2) / 255.0

        # Distribution similarity (cosine)
        features[offset + 3] = self._distribution_similarity(block1, block2)

        # Boundary smoothness (matches C++ - uses 16 bytes)
        features[offset + 4] = self._boundary_smoothness(block1, block2)

        # Cross correlation (matches C++ - uses 64 bytes from boundary)
        features[offset + 5] = self._cross_correlation_cpp(block1, block2)

        # Transition histogram (8 bins, 38-45) - matches C++ implementation
        # C++ computes diff between corresponding bytes and maps to 8 bins
        if len(block1) >= 256 and len(block2) >= 256:
            transition_samples = min(256, min(len(block1), len(block2)))
            transition_count = np.zeros(8, dtype=np.float32)

            for i in range(transition_samples):
                from_byte = block1[len(block1) - transition_samples + i]
                to_byte = block2[i]
                diff = int(to_byte) - int(from_byte)
                # Map diff to 8 bins: bin = (diff + 256) * 8 / 512
                bin_idx = (diff + 256) * 8 // 512
                bin_idx = max(0, min(7, bin_idx))
                transition_count[bin_idx] += 1

            features[offset + 6:offset + 14] = transition_count / transition_samples
        else:
            features[offset + 6:offset + 14] = 0.0

        # Local header at boundary
        features[offset + 14] = self._detect_local_header_boundary(block2)

        # EOCD proximity
        features[offset + 15] = self._detect_eocd_proximity(block1)

    def _extract_zip_features(
        self,
        block1: np.ndarray,
        block2: np.ndarray,
        features: np.ndarray,
        offset: int
    ):
        """Extract 16 ZIP-specific features."""
        e1 = self._calculate_entropy(block1)
        e2 = self._calculate_entropy(block2)

        # DEFLATE continuity
        features[offset + 0] = 1.0 if (e1 > 0.9 and e2 > 0.9) else 0.0

        # Block alignment
        features[offset + 1] = 0.0
        if len(block2) >= 2:
            if block2[0] == 0x50 and block2[1] == 0x4B:
                features[offset + 1] = 1.0

        # Compression ratio
        features[offset + 2] = (e1 + e2) / 2.0

        # ZIP structure score
        features[offset + 3] = self._detect_zip_structure(block2)

        # Central directory signature
        features[offset + 4] = self._detect_central_dir(block2)

        # Data descriptor
        features[offset + 5] = self._detect_data_descriptor(block1)

        # Remaining features
        for i in range(6, 16):
            features[offset + i] = 0.0

    def _extract_mp3_features(
        self,
        block1: np.ndarray,
        block2: np.ndarray,
        features: np.ndarray,
        offset: int
    ):
        """Extract 16 MP3-specific features."""
        # Frame sync detection
        sync1 = self._detect_mp3_sync(block1)
        sync2 = self._detect_mp3_sync(block2)
        features[offset + 0] = min(1.0, (sync1 + sync2) / 20.0)

        # Frame header validity
        features[offset + 1] = self._validate_mp3_header(block2)

        # Bitrate consistency
        features[offset + 2] = 1.0 if (sync1 > 0 and sync2 > 0) else 0.0

        # Sample rate consistency
        features[offset + 3] = features[offset + 2]

        # Frame length consistency
        features[offset + 4] = 1.0 if (sync1 >= 2 or sync2 >= 2) else 0.0

        # ID3 tag detection
        features[offset + 5] = self._detect_id3(block2)

        # VBR header detection
        features[offset + 6] = self._detect_vbr_header(block2)

        # Audio entropy
        e1 = self._calculate_entropy(block1)
        e2 = self._calculate_entropy(block2)
        features[offset + 7] = 1.0 if (e1 > 0.85 and e2 > 0.85) else 0.0

        # Entropy stability
        features[offset + 8] = 1.0 - abs(e1 - e2)

        # Reserved
        for i in range(9, 16):
            features[offset + i] = 0.0

    def _extract_mp4_features(
        self,
        block1: np.ndarray,
        block2: np.ndarray,
        features: np.ndarray,
        offset: int
    ):
        """Extract 16 MP4-specific features."""
        # Box detection
        valid, box_type = self._detect_mp4_box(block2)
        features[offset + 0] = 1.0 if valid else 0.0

        # Box size validity
        features[offset + 1] = self._validate_mp4_box_size(block2)

        # mdat continuity
        e1 = self._calculate_entropy(block1)
        features[offset + 2] = 1.0 if box_type == 'mdat' else (0.7 if e1 > 0.9 else 0.0)

        # moov structure
        features[offset + 3] = 1.0 if box_type in ['moov', 'trak', 'mdia', 'minf', 'stbl'] else 0.0

        # ftyp detection
        features[offset + 4] = 1.0 if box_type == 'ftyp' else 0.0

        # Atom alignment
        features[offset + 5] = 1.0 if valid else 0.0

        # Common box type
        common_boxes = ['ftyp', 'moov', 'mdat', 'free', 'skip', 'wide', 'uuid']
        features[offset + 6] = 1.0 if box_type in common_boxes else 0.0

        # Media data entropy
        e2 = self._calculate_entropy(block2)
        features[offset + 7] = (e1 + e2) / 2.0

        # Entropy stability
        features[offset + 8] = 1.0 - abs(e1 - e2)

        # Reserved
        for i in range(9, 16):
            features[offset + i] = 0.0

    def _extract_generic_features(
        self,
        block1: np.ndarray,
        block2: np.ndarray,
        features: np.ndarray,
        offset: int
    ):
        """Extract 16 generic media features."""
        e1 = self._calculate_entropy(block1)
        e2 = self._calculate_entropy(block2)

        features[offset + 0] = e1 if e1 < 0.9 else 1.0
        features[offset + 1] = e2 if e2 < 0.9 else 1.0
        features[offset + 2] = 1.0 - abs(e1 - e2)
        features[offset + 3] = self._distribution_similarity(block1, block2)
        features[offset + 4] = self._boundary_smoothness(block1, block2)

        # Format detection
        features[offset + 5] = self._detect_riff(block2)
        features[offset + 6] = self._detect_avi(block2)
        features[offset + 7] = self._detect_flac(block2)
        features[offset + 8] = self._detect_ogg(block2)
        features[offset + 9] = self._detect_png(block2)
        features[offset + 10] = self._detect_jpeg(block2)

        for i in range(11, 16):
            features[offset + i] = 0.0

    # Helper methods
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate normalized Shannon entropy."""
        if len(data) == 0:
            return 0.0
        counts = np.bincount(data, minlength=256)
        probs = counts[counts > 0] / len(data)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy / 8.0  # Normalize to [0, 1]

    def _distribution_similarity(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """Calculate cosine similarity of byte distributions."""
        h1 = np.bincount(b1, minlength=256).astype(np.float32)
        h2 = np.bincount(b2, minlength=256).astype(np.float32)
        dot = np.dot(h1, h2)
        norm = np.sqrt(np.dot(h1, h1) * np.dot(h2, h2))
        return dot / norm if norm > 0 else 0.0

    def _boundary_smoothness(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """Calculate boundary smoothness (matches C++ - uses 16 bytes)."""
        if len(b1) < 16 or len(b2) < 16:
            return 0.0
        # C++ uses 16 bytes
        sample_size = min(16, min(len(b1), len(b2)))
        tail = b1[-sample_size:].astype(np.float32)
        head = b2[:sample_size].astype(np.float32)

        total_diff = 0.0
        for i in range(sample_size):
            total_diff += abs(float(tail[i]) - float(head[i]))

        avg_diff = total_diff / sample_size
        return 1.0 - (avg_diff / 255.0)

    def _cross_correlation(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """Calculate cross-correlation at boundary (original method)."""
        if len(b1) < 16 or len(b2) < 16:
            return 0.0
        tail = b1[-16:].astype(np.float32)
        head = b2[:16].astype(np.float32)
        corr = np.corrcoef(tail, head)[0, 1]
        return (corr + 1.0) / 2.0 if not np.isnan(corr) else 0.5

    def _cross_correlation_cpp(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """Calculate cross-correlation (matches C++ implementation - 64 bytes)."""
        boundary_size = min(64, min(len(b1), len(b2)))
        if boundary_size < 1:
            return 0.0

        tail_end = b1[len(b1) - boundary_size:]
        head = b2[:boundary_size]

        corr = 0.0
        for i in range(boundary_size):
            corr += float(tail_end[i]) * float(head[i])

        corr /= (boundary_size * 255.0 * 255.0)
        return float(corr)

    def _detect_pk_signature(self, data: np.ndarray) -> float:
        """Detect PK (ZIP) signature (simple version)."""
        for i in range(len(data) - 1):
            if data[i] == 0x50 and data[i + 1] == 0x4B:
                return 1.0
        return 0.0

    def _detect_pk_signature_strict(self, data: np.ndarray) -> float:
        """Detect valid ZIP signatures (matches C++ implementation)."""
        if len(data) < 4:
            return 0.0

        signature_count = 0
        for i in range(len(data) - 3):
            if data[i] == 0x50 and data[i + 1] == 0x4B:
                sig2 = data[i + 2]
                sig3 = data[i + 3]
                # Check for valid ZIP signatures
                if ((sig2 == 0x03 and sig3 == 0x04) or   # Local file header
                    (sig2 == 0x01 and sig3 == 0x02) or   # Central directory
                    (sig2 == 0x05 and sig3 == 0x06) or   # EOCD
                    (sig2 == 0x07 and sig3 == 0x08)):    # Data descriptor
                    signature_count += 1

        return min(1.0, signature_count / 5.0)

    def _detect_compression_score(self, data: np.ndarray) -> float:
        """Detect compressed data characteristics (matches C++ implementation)."""
        entropy = self._calculate_entropy(data)

        # C++ logic: compressed data entropy is usually 0.9-1.0
        if entropy > 0.9:
            return 1.0
        elif entropy > 0.7:
            return (entropy - 0.7) / 0.2
        return 0.0

    def _detect_local_header_boundary(self, data: np.ndarray) -> float:
        """Detect local file header at block start."""
        if len(data) >= 4:
            if data[0] == 0x50 and data[1] == 0x4B and data[2] == 0x03 and data[3] == 0x04:
                return 1.0
        return 0.0

    def _detect_eocd_proximity(self, data: np.ndarray) -> float:
        """Detect End of Central Directory proximity (matches C++ implementation)."""
        if len(data) < 22:
            return 0.0

        # Search backwards for EOCD signature (up to 65536 bytes back)
        search_limit = min(len(data) - 22, 65535)
        for i in range(len(data) - 22, max(0, len(data) - 22 - search_limit), -1):
            if (data[i] == 0x50 and data[i+1] == 0x4B and
                data[i+2] == 0x05 and data[i+3] == 0x06):
                # Found EOCD, calculate proximity
                proximity = 1.0 - float(len(data) - i) / len(data)
                return proximity
        return 0.0

    def _detect_zip_structure(self, data: np.ndarray) -> float:
        """Detect valid ZIP structure."""
        if len(data) < 4:
            return 0.0
        if data[0] == 0x50 and data[1] == 0x4B:
            if data[2] == 0x03 and data[3] == 0x04:
                return 1.0  # Local file header
            if data[2] == 0x01 and data[3] == 0x02:
                return 0.8  # Central directory
            if data[2] == 0x05 and data[3] == 0x06:
                return 0.6  # EOCD
        return 0.0

    def _detect_central_dir(self, data: np.ndarray) -> float:
        """Detect central directory signature."""
        for i in range(len(data) - 3):
            if (data[i] == 0x50 and data[i+1] == 0x4B and
                data[i+2] == 0x01 and data[i+3] == 0x02):
                return 1.0
        return 0.0

    def _detect_data_descriptor(self, data: np.ndarray) -> float:
        """Detect data descriptor signature."""
        for i in range(len(data) - 3):
            if (data[i] == 0x50 and data[i+1] == 0x4B and
                data[i+2] == 0x07 and data[i+3] == 0x08):
                return 1.0
        return 0.0

    def _detect_mp3_sync(self, data: np.ndarray) -> int:
        """Count MP3 frame syncs."""
        count = 0
        for i in range(len(data) - 1):
            if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                count += 1
        return count

    def _validate_mp3_header(self, data: np.ndarray) -> float:
        """Validate MP3 frame header."""
        if len(data) < 4:
            return 0.0
        if data[0] != 0xFF or (data[1] & 0xE0) != 0xE0:
            return 0.0

        version = (data[1] >> 3) & 0x03
        layer = (data[1] >> 1) & 0x03
        bitrate_idx = (data[2] >> 4) & 0x0F
        sample_rate_idx = (data[2] >> 2) & 0x03

        if version != 1 and layer != 0 and bitrate_idx not in [0, 15] and sample_rate_idx != 3:
            return 1.0
        return 0.0

    def _detect_id3(self, data: np.ndarray) -> float:
        """Detect ID3 tag."""
        if len(data) >= 3:
            if data[0] == ord('I') and data[1] == ord('D') and data[2] == ord('3'):
                return 1.0
        return 0.0

    def _detect_vbr_header(self, data: np.ndarray) -> float:
        """Detect VBR header (Xing/VBRI)."""
        for i in range(len(data) - 3):
            if ((data[i] == ord('X') and data[i+1] == ord('i') and
                 data[i+2] == ord('n') and data[i+3] == ord('g')) or
                (data[i] == ord('V') and data[i+1] == ord('B') and
                 data[i+2] == ord('R') and data[i+3] == ord('I'))):
                return 1.0
        return 0.0

    def _detect_mp4_box(self, data: np.ndarray) -> Tuple[bool, str]:
        """Detect MP4 box and return type."""
        if len(data) < 8:
            return False, ""

        # Read box size (big-endian)
        box_size = (int(data[0]) << 24) | (int(data[1]) << 16) | (int(data[2]) << 8) | int(data[3])

        # Read box type
        try:
            box_type = bytes(data[4:8]).decode('ascii')
            valid_chars = all(c.isalnum() or c in ' -_' for c in box_type)
        except:
            return False, ""

        # Validate size
        valid_size = (8 <= box_size < 0x80000000) or box_size in [0, 1]

        return valid_chars and valid_size, box_type

    def _validate_mp4_box_size(self, data: np.ndarray) -> float:
        """Validate MP4 box size."""
        if len(data) < 8:
            return 0.0
        box_size = (int(data[0]) << 24) | (int(data[1]) << 16) | (int(data[2]) << 8) | int(data[3])
        if 8 <= box_size <= 500 * 1024 * 1024:
            return 1.0
        if box_size in [0, 1]:
            return 0.8
        return 0.0

    def _detect_riff(self, data: np.ndarray) -> float:
        if len(data) >= 4:
            if (data[0] == ord('R') and data[1] == ord('I') and
                data[2] == ord('F') and data[3] == ord('F')):
                return 1.0
        return 0.0

    def _detect_avi(self, data: np.ndarray) -> float:
        if len(data) >= 12:
            if (data[0] == ord('R') and data[1] == ord('I') and
                data[2] == ord('F') and data[3] == ord('F') and
                data[8] == ord('A') and data[9] == ord('V') and
                data[10] == ord('I') and data[11] == ord(' ')):
                return 1.0
        return 0.0

    def _detect_flac(self, data: np.ndarray) -> float:
        if len(data) >= 4:
            if (data[0] == ord('f') and data[1] == ord('L') and
                data[2] == ord('a') and data[3] == ord('C')):
                return 1.0
        return 0.0

    def _detect_ogg(self, data: np.ndarray) -> float:
        if len(data) >= 4:
            if (data[0] == ord('O') and data[1] == ord('g') and
                data[2] == ord('g') and data[3] == ord('S')):
                return 1.0
        return 0.0

    def _detect_png(self, data: np.ndarray) -> float:
        if len(data) >= 4:
            if data[0] == 0x89 and data[1] == ord('P') and data[2] == ord('N') and data[3] == ord('G'):
                return 1.0
        return 0.0

    def _detect_jpeg(self, data: np.ndarray) -> float:
        if len(data) >= 3:
            if data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF:
                return 1.0
        return 0.0

    # High-level API

    def extract_from_file(
        self,
        file_path: str,
        samples_per_file: int = 10,
        min_file_size: int = 32768
    ) -> List[Tuple[np.ndarray, bool]]:
        """
        Extract continuity samples from a file.

        Args:
            file_path: Path to file
            samples_per_file: Number of samples to generate
            min_file_size: Minimum file size (default 32KB)

        Returns:
            List of (features, is_continuous) tuples
        """
        path = Path(file_path)
        file_type = path.suffix.lower().lstrip('.')

        with open(path, 'rb') as f:
            data = f.read()

        if len(data) < min_file_size:
            return []

        samples = []
        num_blocks = len(data) // self.block_size

        if num_blocks < 4:
            return []

        # Generate positive samples (adjacent blocks from same file)
        for _ in range(samples_per_file):
            idx = np.random.randint(0, num_blocks - 1)
            offset1 = idx * self.block_size
            offset2 = (idx + 1) * self.block_size

            block1 = data[offset1:offset1 + self.block_size]
            block2 = data[offset2:offset2 + self.block_size]

            features = self.extract(block1, block2, file_type)
            samples.append((features, True))

        return samples


def extract_dataset_from_directory(
    directory: str,
    output_csv: str,
    file_types: List[str] = None,
    samples_per_file: int = 10,
    max_files: int = 0,
    pos_neg_ratio: float = 1.0
):
    """
    Extract continuity dataset from a directory of files.

    Args:
        directory: Input directory
        output_csv: Output CSV file path
        file_types: List of file extensions to process
        samples_per_file: Samples per file
        max_files: Maximum files to process (0 = unlimited)
        pos_neg_ratio: Ratio of positive to negative samples
    """
    import csv
    from tqdm import tqdm

    if file_types is None:
        file_types = list(ContinuityFeatureExtractor.ZIP_FORMATS |
                          ContinuityFeatureExtractor.AUDIO_FORMATS |
                          ContinuityFeatureExtractor.VIDEO_FORMATS |
                          ContinuityFeatureExtractor.IMAGE_FORMATS)

    extractor = ContinuityFeatureExtractor()
    dir_path = Path(directory)

    # Collect files
    files = []
    for ext in file_types:
        files.extend(dir_path.rglob(f"*.{ext}"))

    if max_files > 0:
        files = files[:max_files]

    print(f"Found {len(files)} files")

    all_samples = []
    file_data_cache = []

    for file_path in tqdm(files, desc="Processing files"):
        try:
            samples = extractor.extract_from_file(str(file_path), samples_per_file)
            for features, is_continuous in samples:
                file_type = file_path.suffix.lower().lstrip('.')
                all_samples.append({
                    'features': features,
                    'is_continuous': int(is_continuous),
                    'file_type': file_type,
                    'sample_type': 'same_file' if is_continuous else 'different_files'
                })

            # Cache for negative samples
            with open(file_path, 'rb') as f:
                data = f.read()
            if len(data) >= 32768:
                file_data_cache.append((file_path, data))
                if len(file_data_cache) > 100:
                    file_data_cache.pop(0)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Generate negative samples (blocks from different files)
    positive_count = sum(1 for s in all_samples if s['is_continuous'])
    negative_needed = int(positive_count * pos_neg_ratio) - sum(1 for s in all_samples if not s['is_continuous'])

    if negative_needed > 0 and len(file_data_cache) >= 2:
        print(f"Generating {negative_needed} negative samples...")
        for _ in range(negative_needed):
            idx1, idx2 = np.random.choice(len(file_data_cache), 2, replace=False)
            path1, data1 = file_data_cache[idx1]
            path2, data2 = file_data_cache[idx2]

            type1 = path1.suffix.lower().lstrip('.')

            # Random offsets
            offset1 = np.random.randint(0, len(data1) - extractor.block_size)
            offset2 = np.random.randint(0, len(data2) - extractor.block_size)

            block1 = data1[offset1:offset1 + extractor.block_size]
            block2 = data2[offset2:offset2 + extractor.block_size]

            features = extractor.extract(block1, block2, type1)
            all_samples.append({
                'features': features,
                'is_continuous': 0,
                'file_type': type1,
                'sample_type': 'different_files'
            })

    # Write CSV
    print(f"Writing {len(all_samples)} samples to {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = [f"f{i}" for i in range(64)] + ['is_continuous', 'file_type', 'sample_type']
        writer.writerow(header)

        # Data
        for sample in all_samples:
            row = list(sample['features']) + [
                sample['is_continuous'],
                sample['file_type'],
                sample['sample_type']
            ]
            writer.writerow(row)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuity Feature Extractor")
    parser.add_argument("directory", help="Input directory")
    parser.add_argument("--output", "-o", default="continuity_dataset.csv", help="Output CSV")
    parser.add_argument("--types", "-t", help="File types (comma-separated)")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Samples per file")
    parser.add_argument("--max-files", "-n", type=int, default=0, help="Max files")

    args = parser.parse_args()

    file_types = args.types.split(',') if args.types else None

    extract_dataset_from_directory(
        directory=args.directory,
        output_csv=args.output,
        file_types=file_types,
        samples_per_file=args.samples,
        max_files=args.max_files
    )
