#pragma once

#include <cstdint>

#include "../common.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "./qg_scanner.hpp"

namespace symqg {
class QGQuery {
   private:
    const float* query_data_ = nullptr;
    std::vector<float, memory::AlignedAllocator<float>> rotated_query_;
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
    std::vector<float, memory::AlignedAllocator<float, 64>> float_lut_;  // 使用float类型的LUT
    size_t padded_dim_ = 0;
    float sum_q_ = 0;
    float width_ = 0;
    float lower_val_ = 0;
    float upper_val_ = 0;


   public:
    explicit QGQuery(const float* q, size_t padded_dim)
        : query_data_(q)
        , rotated_query_(padded_dim)
        , lut_(padded_dim << 2)  // padded_dim / 4 * 16
        , float_lut_(padded_dim << 2)  // padded_dim / 4 * 16，与原来LUT大小一致
        , padded_dim_(padded_dim) {}

    void query_prepare(const FHTRotator& rotator, const QGScanner& scanner) {
        // rotate query (no quantization, keep as float)
        rotator.rotate(query_data_, rotated_query_.data());
        for(size_t i=0; i<padded_dim_; ++i) {
            rotated_query_[i] = rotated_query_[i] * -10;
        }

        for(size_t i=0; i<padded_dim_; ++i) {
            sum_q_ += rotated_query_[i];
        }
        
        // pack float LUT directly from rotated query
        scanner.pack_float_lut(rotated_query_.data(), float_lut_.data());
    }

    [[nodiscard]] const float& width() const { return width_; }

    [[nodiscard]] const float& lower_val() const { return lower_val_; }

    [[nodiscard]] const std::vector<float, memory::AlignedAllocator<float, 64>>& float_lut() const {
        return float_lut_;
    }

    [[nodiscard]] const float& sumq() const { return sum_q_; }

    [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>>& lut(
    ) const {
        return lut_;
    }

    [[nodiscard]] const float* rotated_query_data() const { return rotated_query_.data(); }

    [[nodiscard]] const float* query_data() const { return query_data_; }
};
}  // namespace symqg