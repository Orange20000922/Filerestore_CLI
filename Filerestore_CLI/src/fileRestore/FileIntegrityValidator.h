#pragma once
#include <Windows.h>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

// ============================================================================
// File Integrity Validation Module
//
// Purpose: Detect corrupted or damaged files during signature-based recovery
// Methods: Entropy analysis, structure validation, statistical analysis
// ============================================================================

// Validation result for a single file
struct FileIntegrityScore {
    double entropyScore;        // Entropy analysis score (0-1)
    double structureScore;      // Structure validation score (0-1)
    double statisticalScore;    // Statistical analysis score (0-1)
    double footerScore;         // Footer validation score (0-1)
    double overallScore;        // Combined overall score (0-1)

    string diagnosis;           // Human-readable diagnosis
    bool isLikelyCorrupted;     // Quick check result

    // Detailed info
    double entropy;             // Raw entropy value (0-8)
    double zeroRatio;           // Zero byte ratio (0-1)
    double chiSquare;           // Chi-square value
    bool hasValidHeader;        // Header validation result
    bool hasValidFooter;        // Footer validation result
    bool hasEntropyAnomaly;     // Entropy anomaly detected
    size_t anomalyOffset;       // Offset where anomaly detected (if any)

    FileIntegrityScore() : entropyScore(0), structureScore(0), statisticalScore(0),
                           footerScore(0), overallScore(0), isLikelyCorrupted(false),
                           entropy(0), zeroRatio(0), chiSquare(0),
                           hasValidHeader(false), hasValidFooter(false),
                           hasEntropyAnomaly(false), anomalyOffset(0) {}
};

// Entropy characteristics for different file types
struct EntropyProfile {
    string extension;
    double expectedMin;         // Minimum expected entropy
    double expectedMax;         // Maximum expected entropy
    double anomalyThreshold;    // Entropy variance threshold for anomaly
    bool isCompressed;          // Whether file is typically compressed
};

// JPEG structure validation result
struct JPEGValidation {
    bool hasSOI;                // Start of Image (FFD8)
    bool hasEOI;                // End of Image (FFD9)
    bool hasValidMarkers;       // Valid segment markers
    bool hasValidDHT;           // Huffman table
    bool hasValidDQT;           // Quantization table
    bool hasSOF;                // Start of Frame
    bool hasSOS;                // Start of Scan
    int markerCount;            // Total marker count
    double confidence;          // Overall confidence
};

// PNG structure validation result
struct PNGValidation {
    bool hasValidSignature;     // 8-byte PNG signature
    bool hasIHDR;               // Image Header chunk
    bool hasIEND;               // Image End chunk
    bool hasValidCRC;           // CRC checks passed
    int chunkCount;             // Total chunk count
    double confidence;          // Overall confidence
};

// ZIP structure validation result
struct ZIPValidation {
    bool hasValidLocalHeader;   // Local file header (PK..)
    bool hasValidCentralDir;    // Central directory
    bool hasEndOfCentralDir;    // End of central directory
    DWORD declaredFileCount;    // Declared file count
    DWORD actualFileCount;      // Actual file count found
    double confidence;          // Overall confidence
};

// PDF structure validation result
struct PDFValidation {
    bool hasValidHeader;        // %PDF-x.x header
    bool hasEOF;                // %%EOF marker
    bool hasXRef;               // Cross-reference table
    bool hasTrailer;            // Trailer dictionary
    int objectCount;            // Object count found
    double confidence;          // Overall confidence
};

class FileIntegrityValidator {
private:
    // Entropy profiles for different file types
    static const EntropyProfile entropyProfiles[];
    static const int entropyProfileCount;

    // Get entropy profile for file type
    static const EntropyProfile* GetEntropyProfile(const string& extension);

    // Entropy calculation helpers
    static double CalculateEntropy(const BYTE* data, size_t size);
    static vector<double> CalculateEntropyBlocks(const BYTE* data, size_t size, size_t blockSize);
    static bool DetectEntropyAnomaly(const vector<double>& entropies, double threshold, size_t& anomalyOffset);

    // Statistical analysis helpers
    static double CalculateChiSquare(const BYTE* data, size_t size);
    static double CalculateZeroRatio(const BYTE* data, size_t size);
    static bool IsLikelyRandom(const BYTE* data, size_t size);

    // Structure validation for specific formats
    static JPEGValidation ValidateJPEG(const BYTE* data, size_t size);
    static PNGValidation ValidatePNG(const BYTE* data, size_t size);
    static ZIPValidation ValidateZIP(const BYTE* data, size_t size);
    static PDFValidation ValidatePDF(const BYTE* data, size_t size);

    // Score calculation helpers
    static double EvaluateEntropyForType(double entropy, const string& extension);
    static double EvaluateStatistics(double zeroRatio, double chiSquare, const string& extension);
    static double ValidateFooter(const BYTE* data, size_t size, const string& extension);
    static double ValidateFileStructure(const BYTE* data, size_t size, const string& extension);

    // CRC32 calculation for PNG validation
    static DWORD CalculateCRC32(const BYTE* data, size_t size);
    static DWORD ReadBigEndian32(const BYTE* data);

public:
    // Main validation function
    static FileIntegrityScore Validate(const BYTE* data, size_t size, const string& extension);

    // Quick check for likely corruption
    static bool IsLikelyCorrupted(const BYTE* data, size_t size, const string& extension);

    // Individual component scores
    static double GetEntropyScore(const BYTE* data, size_t size, const string& extension);
    static double GetStructureScore(const BYTE* data, size_t size, const string& extension);
    static double GetStatisticalScore(const BYTE* data, size_t size, const string& extension);

    // Thresholds
    static constexpr double MIN_INTEGRITY_SCORE = 0.5;      // Minimum acceptable score
    static constexpr double HIGH_CONFIDENCE_SCORE = 0.8;    // High confidence threshold
    static constexpr double ENTROPY_VARIANCE_THRESHOLD = 1.5; // Entropy anomaly threshold
    static constexpr double MAX_ZERO_RATIO_COMPRESSED = 0.05; // Max zero ratio for compressed
    static constexpr double MAX_ZERO_RATIO_GENERAL = 0.15;    // Max zero ratio general
};
