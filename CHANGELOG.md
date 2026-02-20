# Changelog

All notable changes to Filerestore_CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-02-20

### Added
- USN targeted recovery system (`usnlist`, `usnrecover`, `recover` commands)
- MFT snapshot storage — captures complete metadata at the moment of file deletion
- USN delete monitor background daemon with real-time event tracking
- Monitor daemon manager with shared memory IPC and Windows auto-start
- Kernel driver bridge client (experimental, minifilter communication, disabled by default)
- **Centralized bad cluster filtered reader** (`ClusterFilteredReader`) — per-cluster overwrite detection and filtering at data read stage, unified across all 3 recovery paths (USN targeted, signature carving, API)
- Cluster health report output: health percentage, detection time, truncation info
- Format-aware truncation for PNG (IEND), ZIP (EOCD), JPEG (FFD9), PDF (%%EOF) with 50% safety threshold
- FileCarver progress sync to TUI for all 6 scan functions

### Improved
- All recovery paths (USN, carving, API) now use unified cluster filtering
- MFT cache v2: sequence number validation, global singleton `MFTCacheManager`, auto-expiry
- TUI multi-view modes: welcome page, parameter forms, scan progress, results table
- UsnTargetedRecovery: size/extension static filters, batch operations, MFT enrichment
- MFTReader: optimized cluster read performance

### Fixed
- JPEG format truncation used forward search, incorrectly truncating to embedded thumbnail's FFD9; now uses reverse search
- Multiple known issues

### Refactored
- Unified command registration architecture, removed deprecated `cmd.cpp`

## [0.3.2] - 2026-02-07

### Added
- Modern TUI interface (FTXUI framework)
- Google Test unit testing (45 tests)
- SIMD signature matching optimization (SSE2/AVX2, +8% throughput)
- `--cmd` option for non-interactive command execution (CI/CD automation)
- GitHub Actions CI/CD pipeline with FTXUI/ONNX caching

### Improved
- Dependency management documentation

## [0.3.1] - 2026-01-07

### Added
- `crp` interactive paged recovery command with batch operations

## [0.3.0] - 2026-01-07

### Added
- ML file classification via ONNX Runtime (19 file types)
- Hybrid scanning mode: auto-select signature or ML based on file type
- Threaded signature scanning with configurable thread count

[1.0.0]: https://github.com/Orange20000922/Filerestore_CLI/compare/v0.3.2...v1.0.0
[0.3.2]: https://github.com/Orange20000922/Filerestore_CLI/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Orange20000922/Filerestore_CLI/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Orange20000922/Filerestore_CLI/releases/tag/v0.3.0
