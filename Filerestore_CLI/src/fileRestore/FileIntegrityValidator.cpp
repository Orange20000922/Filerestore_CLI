#include "FileIntegrityValidator.h"
#include <algorithm>
#include <cstring>

// ============================================================================
// Entropy Profiles for Different File Types
// ============================================================================
const EntropyProfile FileIntegrityValidator::entropyProfiles[] = {
    {"jpg",    7.0, 7.95, 1.5, true},   // JPEG: high entropy, compressed
    {"jpeg",   7.0, 7.95, 1.5, true},
    {"png",    6.5, 7.9,  1.5, true},   // PNG: high entropy, compressed
    {"gif",    5.0, 7.5,  1.5, true},   // GIF: variable, LZW compressed
    {"pdf",    4.0, 7.8,  2.0, false},  // PDF: mixed content
    {"zip",    7.2, 7.98, 1.0, true},   // ZIP: very high entropy
    {"7z",     7.5, 7.99, 0.8, true},   // 7z: extremely high entropy
    {"rar",    7.3, 7.98, 1.0, true},   // RAR: very high entropy
    {"mp3",    6.5, 7.9,  1.5, true},   // MP3: compressed audio
    {"mp4",    6.0, 7.9,  1.5, true},   // MP4: compressed video
    {"avi",    5.0, 7.5,  2.0, false},  // AVI: variable compression
    {"wav",    3.0, 7.0,  2.5, false},  // WAV: often uncompressed
    {"exe",    5.0, 7.5,  2.0, false},  // EXE: mixed code and data
    {"dll",    5.0, 7.5,  2.0, false},
    {"bmp",    2.0, 6.0,  2.5, false},  // BMP: usually uncompressed
    {"sqlite", 4.0, 7.5,  2.0, false},  // SQLite: database
};

const int FileIntegrityValidator::entropyProfileCount = sizeof(entropyProfiles) / sizeof(EntropyProfile);

// ============================================================================
// Get Entropy Profile
// ============================================================================
const EntropyProfile* FileIntegrityValidator::GetEntropyProfile(const string& extension) {
    string ext = extension;
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    for (int i = 0; i < entropyProfileCount; i++) {
        if (entropyProfiles[i].extension == ext) {
            return &entropyProfiles[i];
        }
    }
    return nullptr;
}

// ============================================================================
// Shannon Entropy Calculation
// ============================================================================
double FileIntegrityValidator::CalculateEntropy(const BYTE* data, size_t size) {
    if (size == 0) return 0.0;

    int frequency[256] = {0};
    for (size_t i = 0; i < size; i++) {
        frequency[data[i]]++;
    }

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            double p = (double)frequency[i] / size;
            entropy -= p * log2(p);
        }
    }
    return entropy;  // 0-8 bits per byte
}

// ============================================================================
// Block-wise Entropy Calculation (for anomaly detection)
// ============================================================================
vector<double> FileIntegrityValidator::CalculateEntropyBlocks(const BYTE* data, size_t size, size_t blockSize) {
    vector<double> entropies;
    for (size_t i = 0; i < size; i += blockSize) {
        size_t len = min(blockSize, size - i);
        entropies.push_back(CalculateEntropy(data + i, len));
    }
    return entropies;
}

// ============================================================================
// Detect Entropy Anomaly (sudden changes indicating corruption)
// ============================================================================
bool FileIntegrityValidator::DetectEntropyAnomaly(const vector<double>& entropies,
                                                   double threshold, size_t& anomalyOffset) {
    if (entropies.size() < 2) return false;

    for (size_t i = 1; i < entropies.size(); i++) {
        double diff = abs(entropies[i] - entropies[i-1]);
        if (diff > threshold) {
            anomalyOffset = i;
            return true;
        }
    }
    return false;
}

double FileIntegrityValidator::CalculateChiSquare(const BYTE* data, size_t size) {
    if (size == 0) return 0.0;

    int observed[256] = {0};
    double expected = (double)size / 256.0;

    for (size_t i = 0; i < size; i++) {
        observed[data[i]]++;
    }

    double chiSquare = 0.0;
    for (int i = 0; i < 256; i++) {
        double diff = observed[i] - expected;
        chiSquare += (diff * diff) / expected;
    }

    return chiSquare;
}

// ============================================================================
// Zero Byte Ratio Calculation
// ============================================================================
double FileIntegrityValidator::CalculateZeroRatio(const BYTE* data, size_t size) {
    if (size == 0) return 0.0;

    size_t zeroCount = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == 0) zeroCount++;
    }
    return (double)zeroCount / size;
}

bool FileIntegrityValidator::IsLikelyRandom(const BYTE* data, size_t size) {
    double chi = CalculateChiSquare(data, min(size, (size_t)65536));
    // Chi-square critical value for df=255, p=0.05 is ~293
    return chi < 350;
}
static const DWORD crcTable[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
    0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
    0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
    0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
    0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
    0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
    0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
    0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
    0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
    0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
    0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
    0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
    0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
    0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
    0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
    0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
    0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
    0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
    0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
    0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
    0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55,
    0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
    0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28,
    0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
    0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
    0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
    0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
    0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
    0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
    0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD706B3,
    0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
};

DWORD FileIntegrityValidator::CalculateCRC32(const BYTE* data, size_t size) {
    DWORD crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; i++) {
        crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

DWORD FileIntegrityValidator::ReadBigEndian32(const BYTE* data) {
    return ((DWORD)data[0] << 24) | ((DWORD)data[1] << 16) |
           ((DWORD)data[2] << 8) | data[3];
}

// ============================================================================
// JPEG Structure Validation
// ============================================================================
JPEGValidation FileIntegrityValidator::ValidateJPEG(const BYTE* data, size_t size) {
    JPEGValidation result = {0};

    if (size < 2 || data[0] != 0xFF || data[1] != 0xD8) {
        return result;
    }
    result.hasSOI = true;

    size_t pos = 2;
    while (pos + 2 < size) {
        if (data[pos] != 0xFF) {
            pos++;
            continue;
        }

        BYTE marker = data[pos + 1];

        // Skip padding FF bytes
        if (marker == 0xFF) {
            pos++;
            continue;
        }

        // EOI - End of Image
        if (marker == 0xD9) {
            result.hasEOI = true;
            result.hasValidMarkers = true;
            break;
        }

        // Check key markers
        if (marker == 0xC4) result.hasValidDHT = true;  // Huffman table
        if (marker == 0xDB) result.hasValidDQT = true;  // Quantization table
        if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
            result.hasSOF = true;  // Start of Frame
        }
        if (marker == 0xDA) result.hasSOS = true;  // Start of Scan

        result.markerCount++;

        // Skip segment
        if (marker >= 0xD0 && marker <= 0xD9) {
            pos += 2;  // Standalone markers
        } else if (pos + 4 < size) {
            WORD segLen = ((WORD)data[pos + 2] << 8) | data[pos + 3];
            pos += 2 + segLen;
        } else {
            break;
        }
    }

    // Calculate confidence
    result.confidence = 0.0;
    if (result.hasSOI) result.confidence += 0.2;
    if (result.hasEOI) result.confidence += 0.3;
    if (result.hasValidDHT) result.confidence += 0.1;
    if (result.hasValidDQT) result.confidence += 0.1;
    if (result.hasSOF) result.confidence += 0.15;
    if (result.hasSOS) result.confidence += 0.15;

    return result;
}

// ============================================================================
// PNG Structure Validation
// ============================================================================
PNGValidation FileIntegrityValidator::ValidatePNG(const BYTE* data, size_t size) {
    PNGValidation result = {0};

    // Check PNG signature
    const BYTE pngSig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (size < 8 || memcmp(data, pngSig, 8) != 0) {
        return result;
    }
    result.hasValidSignature = true;

    // Parse chunks
    size_t pos = 8;
    while (pos + 12 <= size) {
        DWORD length = ReadBigEndian32(data + pos);
        const BYTE* chunkType = data + pos + 4;

        if (length > size - pos - 12) {
            break;  // Invalid chunk length
        }

        // Check chunk type
        if (memcmp(chunkType, "IHDR", 4) == 0) {
            result.hasIHDR = true;
        } else if (memcmp(chunkType, "IEND", 4) == 0) {
            result.hasIEND = true;
        }

        // Validate CRC (for first few chunks only to save time)
        if (result.chunkCount < 5) {
            const BYTE* chunkData = data + pos + 4;  // type + data
            size_t chunkDataLen = 4 + length;
            DWORD calculatedCRC = CalculateCRC32(chunkData, chunkDataLen);
            DWORD storedCRC = ReadBigEndian32(data + pos + 8 + length);
            if (calculatedCRC == storedCRC) {
                result.hasValidCRC = true;
            }
        }

        result.chunkCount++;
        pos += 12 + length;  // length(4) + type(4) + data + crc(4)
    }

    // Calculate confidence
    result.confidence = 0.0;
    if (result.hasValidSignature) result.confidence += 0.3;
    if (result.hasIHDR) result.confidence += 0.2;
    if (result.hasIEND) result.confidence += 0.3;
    if (result.hasValidCRC) result.confidence += 0.2;

    return result;
}

// ============================================================================
// ZIP Structure Validation
// ============================================================================
ZIPValidation FileIntegrityValidator::ValidateZIP(const BYTE* data, size_t size) {
    ZIPValidation result = {0};

    // Check Local File Header
    if (size < 30 || data[0] != 0x50 || data[1] != 0x4B ||
        data[2] != 0x03 || data[3] != 0x04) {
        return result;
    }
    result.hasValidLocalHeader = true;

    // Count local file headers
    size_t pos = 0;
    while (pos + 30 < size) {
        if (data[pos] == 0x50 && data[pos + 1] == 0x4B) {
            if (data[pos + 2] == 0x03 && data[pos + 3] == 0x04) {
                result.actualFileCount++;
                // Skip to next header
                WORD nameLen = *(WORD*)(data + pos + 26);
                WORD extraLen = *(WORD*)(data + pos + 28);
                DWORD compSize = *(DWORD*)(data + pos + 18);
                pos += 30 + nameLen + extraLen + compSize;
            } else if (data[pos + 2] == 0x01 && data[pos + 3] == 0x02) {
                result.hasValidCentralDir = true;
                break;
            } else {
                pos++;
            }
        } else {
            pos++;
        }
    }

    // Search for End of Central Directory (from end)
    for (size_t i = size - 22; i > 0 && i > size - 65536; i--) {
        if (data[i] == 0x50 && data[i + 1] == 0x4B &&
            data[i + 2] == 0x05 && data[i + 3] == 0x06) {
            result.hasEndOfCentralDir = true;
            result.declaredFileCount = *(WORD*)(data + i + 8);
            break;
        }
    }

    // Calculate confidence
    result.confidence = 0.3;  // Valid header
    if (result.hasValidCentralDir) result.confidence += 0.2;
    if (result.hasEndOfCentralDir) result.confidence += 0.3;
    if (result.actualFileCount > 0) result.confidence += 0.2;

    return result;
}

// ============================================================================
// PDF Structure Validation
// ============================================================================
PDFValidation FileIntegrityValidator::ValidatePDF(const BYTE* data, size_t size) {
    PDFValidation result = {0};

    // Check PDF header
    if (size < 8 || memcmp(data, "%PDF-", 5) != 0) {
        return result;
    }
    result.hasValidHeader = true;

    // Search for %%EOF (from end)
    const char* eofMarker = "%%EOF";
    size_t eofLen = strlen(eofMarker);
    for (size_t i = size - eofLen; i > size - 1024 && i > 0; i--) {
        if (memcmp(data + i, eofMarker, eofLen) == 0) {
            result.hasEOF = true;
            break;
        }
    }

    // Search for xref
    const char* xrefMarker = "xref";
    for (size_t i = 0; i < size - 4 && i < 100000; i++) {
        if (memcmp(data + i, xrefMarker, 4) == 0) {
            result.hasXRef = true;
            break;
        }
    }

    // Search for trailer
    const char* trailerMarker = "trailer";
    for (size_t i = size > 10000 ? size - 10000 : 0; i < size - 7; i++) {
        if (memcmp(data + i, trailerMarker, 7) == 0) {
            result.hasTrailer = true;
            break;
        }
    }

    // Count objects (rough estimate)
    const char* objMarker = " obj";
    for (size_t i = 0; i < size - 4; i++) {
        if (memcmp(data + i, objMarker, 4) == 0) {
            result.objectCount++;
        }
    }

    // Calculate confidence
    result.confidence = 0.0;
    if (result.hasValidHeader) result.confidence += 0.25;
    if (result.hasEOF) result.confidence += 0.25;
    if (result.hasXRef) result.confidence += 0.2;
    if (result.hasTrailer) result.confidence += 0.2;
    if (result.objectCount > 0) result.confidence += 0.1;

    return result;
}

// ============================================================================
// Evaluate Entropy Score for File Type
// ============================================================================
double FileIntegrityValidator::EvaluateEntropyForType(double entropy, const string& extension) {
    const EntropyProfile* profile = GetEntropyProfile(extension);

    if (!profile) {
        // Default evaluation for unknown types
        if (entropy >= 3.0 && entropy <= 7.8) {
            return 0.7;
        }
        return 0.5;
    }

    // Check if entropy is within expected range
    if (entropy >= profile->expectedMin && entropy <= profile->expectedMax) {
        return 1.0;
    }

    // Penalize deviation
    double deviation = 0;
    if (entropy < profile->expectedMin) {
        deviation = profile->expectedMin - entropy;
    } else {
        deviation = entropy - profile->expectedMax;
    }

    // Score decreases with deviation
    double score = max(0.0, 1.0 - deviation * 0.2);
    return score;
}

// ============================================================================
// Evaluate Statistical Score
// ============================================================================
double FileIntegrityValidator::EvaluateStatistics(double zeroRatio, double chiSquare,
                                                   const string& extension) {
    double score = 1.0;
    const EntropyProfile* profile = GetEntropyProfile(extension);

    // Check zero ratio
    double maxZeroRatio = (profile && profile->isCompressed) ?
                           MAX_ZERO_RATIO_COMPRESSED : MAX_ZERO_RATIO_GENERAL;

    if (zeroRatio > maxZeroRatio) {
        double excess = zeroRatio - maxZeroRatio;
        score -= excess * 2.0;  // Penalize heavily
    }

    // Check chi-square (for compressed files, should be low)
    if (profile && profile->isCompressed) {
        if (chiSquare > 500) {
            score -= 0.2;  // Not random enough for compressed data
        }
    }

    return max(0.0, min(1.0, score));
}

// ============================================================================
// Validate File Structure
// ============================================================================
double FileIntegrityValidator::ValidateFileStructure(const BYTE* data, size_t size,
                                                      const string& extension) {
    string ext = extension;
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "jpg" || ext == "jpeg") {
        JPEGValidation v = ValidateJPEG(data, size);
        return v.confidence;
    }
    if (ext == "png") {
        PNGValidation v = ValidatePNG(data, size);
        return v.confidence;
    }
    if (ext == "zip" || ext == "docx" || ext == "xlsx" || ext == "pptx") {
        ZIPValidation v = ValidateZIP(data, size);
        return v.confidence;
    }
    if (ext == "pdf") {
        PDFValidation v = ValidatePDF(data, size);
        return v.confidence;
    }

    // Default: check basic structure
    if (size < 10) return 0.3;

    // Check for obvious corruption (large zero blocks at start)
    size_t zeroStart = 0;
    for (size_t i = 0; i < min(size, (size_t)1024); i++) {
        if (data[i] == 0) zeroStart++;
        else break;
    }
    if (zeroStart > 100) return 0.3;

    return 0.6;  // Neutral score for unknown formats
}

// ============================================================================
// Validate Footer
// ============================================================================
double FileIntegrityValidator::ValidateFooter(const BYTE* data, size_t size,
                                               const string& extension) {
    string ext = extension;
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // JPEG EOI
    if (ext == "jpg" || ext == "jpeg") {
        // Search for FFD9 near end
        for (size_t i = size > 100 ? size - 100 : 0; i < size - 1; i++) {
            if (data[i] == 0xFF && data[i + 1] == 0xD9) {
                return 1.0;
            }
        }
        return 0.3;
    }

    // PNG IEND
    if (ext == "png") {
        const BYTE iend[] = {0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};
        for (size_t i = size > 20 ? size - 20 : 0; i < size - 8; i++) {
            if (memcmp(data + i, iend, 8) == 0) {
                return 1.0;
            }
        }
        return 0.3;
    }

    // PDF %%EOF
    if (ext == "pdf") {
        const char* eof = "%%EOF";
        for (size_t i = size > 1024 ? size - 1024 : 0; i < size - 5; i++) {
            if (memcmp(data + i, eof, 5) == 0) {
                return 1.0;
            }
        }
        return 0.3;
    }

    // ZIP End of Central Directory
    if (ext == "zip") {
        for (size_t i = size > 65536 ? size - 65536 : 0; i < size - 4; i++) {
            if (data[i] == 0x50 && data[i + 1] == 0x4B &&
                data[i + 2] == 0x05 && data[i + 3] == 0x06) {
                return 1.0;
            }
        }
        return 0.4;
    }

    // Default: no specific footer expected
    return 0.7;
}

// ============================================================================
// Main Validation Function
// ============================================================================
FileIntegrityScore FileIntegrityValidator::Validate(const BYTE* data, size_t size,
                                                     const string& extension) {
    FileIntegrityScore score;

    if (size == 0) {
        score.diagnosis = "Empty file";
        score.isLikelyCorrupted = true;
        return score;
    }

    // 1. Entropy Analysis (weight: 25%)
    score.entropy = CalculateEntropy(data, min(size, (size_t)1048576));  // Max 1MB
    score.entropyScore = EvaluateEntropyForType(score.entropy, extension);

    // Check for entropy anomalies
    if (size > 8192) {
        vector<double> blockEntropies = CalculateEntropyBlocks(data, min(size, (size_t)1048576), 4096);
        score.hasEntropyAnomaly = DetectEntropyAnomaly(blockEntropies, ENTROPY_VARIANCE_THRESHOLD,
                                                        score.anomalyOffset);
        if (score.hasEntropyAnomaly) {
            score.entropyScore *= 0.7;  // Penalize anomaly
        }
    }

    // 2. Structure Validation (weight: 35%)
    score.structureScore = ValidateFileStructure(data, size, extension);
    score.hasValidHeader = (score.structureScore > 0.5);

    // 3. Statistical Analysis (weight: 20%)
    size_t statSize = min(size, (size_t)65536);
    score.zeroRatio = CalculateZeroRatio(data, statSize);
    score.chiSquare = CalculateChiSquare(data, statSize);
    score.statisticalScore = EvaluateStatistics(score.zeroRatio, score.chiSquare, extension);

    // 4. Footer Validation (weight: 20%)
    score.footerScore = ValidateFooter(data, size, extension);
    score.hasValidFooter = (score.footerScore > 0.8);

    // Calculate overall score
    score.overallScore =
        score.entropyScore * 0.25 +
        score.structureScore * 0.35 +
        score.statisticalScore * 0.20 +
        score.footerScore * 0.20;

    // Generate diagnosis
    if (score.overallScore >= HIGH_CONFIDENCE_SCORE) {
        score.diagnosis = "High confidence - likely intact";
        score.isLikelyCorrupted = false;
    } else if (score.overallScore >= 0.6) {
        score.diagnosis = "Medium confidence - may have minor issues";
        score.isLikelyCorrupted = false;
    } else if (score.overallScore >= MIN_INTEGRITY_SCORE) {
        score.diagnosis = "Low confidence - likely damaged";
        score.isLikelyCorrupted = true;
    } else {
        score.diagnosis = "Very low confidence - probably corrupted";
        score.isLikelyCorrupted = true;
    }

    // Add details to diagnosis
    if (score.hasEntropyAnomaly) {
        score.diagnosis += " [Entropy anomaly detected]";
    }
    if (!score.hasValidFooter && (extension == "jpg" || extension == "png" || extension == "pdf")) {
        score.diagnosis += " [Missing footer]";
    }
    if (score.zeroRatio > 0.2) {
        score.diagnosis += " [High zero ratio]";
    }

    return score;
}

// ============================================================================
// Quick Corruption Check
// ============================================================================
bool FileIntegrityValidator::IsLikelyCorrupted(const BYTE* data, size_t size,
                                                const string& extension) {
    FileIntegrityScore score = Validate(data, size, extension);
    return score.isLikelyCorrupted;
}

// ============================================================================
// Individual Score Functions
// ============================================================================
double FileIntegrityValidator::GetEntropyScore(const BYTE* data, size_t size,
                                                const string& extension) {
    double entropy = CalculateEntropy(data, min(size, (size_t)1048576));
    return EvaluateEntropyForType(entropy, extension);
}

double FileIntegrityValidator::GetStructureScore(const BYTE* data, size_t size,
                                                  const string& extension) {
    return ValidateFileStructure(data, size, extension);
}

double FileIntegrityValidator::GetStatisticalScore(const BYTE* data, size_t size,
                                                    const string& extension) {
    size_t statSize = min(size, (size_t)65536);
    double zeroRatio = CalculateZeroRatio(data, statSize);
    double chiSquare = CalculateChiSquare(data, statSize);
    return EvaluateStatistics(zeroRatio, chiSquare, extension);
}
