#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace symqg::space {

inline auto popcount(size_t dim, const uint64_t* __restrict__ data) -> size_t {
    size_t ret = 0;
    for (size_t i = 0; i < dim / 64; ++i) {
        ret += __builtin_popcountll((*data));
        ++data;
    }
    return ret;
}

/* Change 0/1 data to uint64 */
inline void pack_binary(
    const int* __restrict__ bin_x, uint64_t* __restrict__ binary, size_t length
) {
    for (size_t i = 0; i < length; i += 64) {
        uint64_t cur = 0;
        for (size_t j = 0; j < 64; ++j) {
            cur |= (static_cast<uint64_t>(bin_x[i + j]) << (63 - j));
        }
        *binary = cur;
        ++binary;
    }
}

/* 改进版本：直接产生用于accumulate的正确字节序 */
inline void pack_binary_ordered(
    const int* __restrict__ bin_x, uint64_t* __restrict__ ordered_binary, size_t length
) {
    for (size_t i = 0; i < length; i += 64) {
        uint64_t cur = 0;
        for (size_t j = 0; j < 64; ++j) {
            cur |= (static_cast<uint64_t>(bin_x[i + j]) << (63 - j));
        }
        
        // 应用字节反序和4位交换，产生可直接用于accumulate的格式
        uint8_t bytes[8];
        for (int b = 0; b < 8; ++b) {
            bytes[b] = (cur >> (b * 8)) & 0xFF;
        }
        
        // 字节序反序 (前4字节与后4字节交换)
        for (int k = 0; k < 4; ++k) {
            std::swap(bytes[k], bytes[7 - k]);
        }
        
        // 4位交换 (每个字节内的高4位和低4位交换)
        for (int b = 0; b < 8; ++b) {
            uint8_t val = bytes[b];
            uint8_t val_hi = (val >> 4);
            uint8_t val_lo = (val & 15);
            bytes[b] = (val_lo << 4) | val_hi;
        }
        
        // 重新组装成uint64_t
        uint64_t ordered_code = 0;
        for (int b = 0; b < 8; ++b) {
            ordered_code |= (static_cast<uint64_t>(bytes[b]) << (b * 8));
        }
        
        *ordered_binary = ordered_code;
        ++ordered_binary;
    }
}

}  // namespace symqg::space