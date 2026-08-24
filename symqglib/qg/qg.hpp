#pragma once

#include <omp.h>

#include <cassert>
#include <cfloat>  // 添加这个来解决FLT_MAX问题
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

#include "../common.hpp"
#include "../quantization/rabitq.hpp"
#include "../space/l2.hpp"
#include "../third/ngt/hashset.hpp"
#include "../third/svs/array.hpp"
#include "../utils/buffer.hpp"
#include "../utils/io.hpp"
#include "../utils/memory.hpp"
#include "../utils/rotator.hpp"
#include "./qg_query.hpp"
#include "./qg_scanner.hpp"
// #define DEBUG

namespace symqg {

class TimerProfile {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    TimerProfile() = default;
    
    void start(const std::string& name) {
        // 记录开始时间
        auto now = std::chrono::high_resolution_clock::now();
        
        // 创建新的计时记录
        active_timers_.push_back({name, now});
    }
    
    void stop() {
        if (active_timers_.empty()) {
            return;
        }
        
        // 获取当前时间
        auto now = std::chrono::high_resolution_clock::now();
        
        // 取出最近启动的计时器
        auto& last_timer = active_timers_.back();
        std::string timer_name = last_timer.name;
        auto start_time = last_timer.start_time;
        
        // 计算经过的时间
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - start_time).count();
        
        // 累加到对应的计时器
        timers_[timer_name] += elapsed;
        
        // 移除这个计时器
        active_timers_.pop_back();
    }
    
    void report() const {
        std::cout << "=== Search Time Breakdown ===" << std::endl;
        for (const auto& timer : timers_) {
            std::cout << timer.first << ": " << timer.second << " us" << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }
    
    void reset() {
        timers_.clear();
        active_timers_.clear();
    }
    
private:
    struct ActiveTimer {
        std::string name;
        TimePoint start_time;
    };
    
    std::unordered_map<std::string, int64_t> timers_;  // 累计时间
    std::vector<ActiveTimer> active_timers_;           // 活动计时器栈
};


/**
 * @brief this Factor only for illustration, the true storage is continous
 * 现在只存储 degree_bound_*triple_x (factor_dq 和 factor_vq 已被移除)
 *
 */
struct Factor {
    float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0
    // float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q) - 不再使用
    // float factor_vq;  // Factor of v_l * ||q_r|| - 不再使用
};

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;    // num points
    size_t degree_bound_ = 0;  // degree bound
    size_t dimension_ = 0;     // dimension
    size_t padded_dim_ = 0;    // padded dimension
    PID entry_point_ = 0;      // Entry point of graph
    mutable TimerProfile timer_;
    bool enable_profiling_ = false;
    

    data::Array<
    float,
    std::vector<size_t>,
    memory::AlignedAllocator<
        float,
        64,
        false>>
    data_;  // graph 
    
    data::Array<
    float,
    std::vector<size_t>,
    memory::AlignedAllocator<
        float,
        1 << 22,
        true>>
    original_data_;  // vectors

    data::Array<
        float,                         // 元素类型
        std::vector<size_t>,           // 形状描述容器
        memory::AlignedAllocator<      // 使用相同的内存分配器
            float,
            64,                        // 使用标准的64字节对齐（或根据需要更改）
            false                      // 不使用大页内存（或根据需要设为true）
    >> centroids_;

    data::Array<
        uint64_t,  // 使用 uint8_t 作为元素类型
        std::vector<size_t>,
        memory::AlignedAllocator<
            uint64_t,
            128,    // 64字节对齐应该足够
            false  // 视情况可以调整为 true
        >
    > packed_codes_;  // 存储所有节点的 packed code

    data::Array<
        float,  // 元素类型
        std::vector<size_t>,
        memory::AlignedAllocator<
            float,
            64,    // 64字节对齐应该足够
            false  // 视情况可以调整为 true
        >
    > factor_codes_;  // 存储所有节点的 packed code

    QGScanner scanner_;
    FHTRotator rotator_;
    HashBasedBooleanSet visited_;
    HashBasedBooleanSet visited_packed_;
    buffer::SearchBuffer search_pool_;

    /*
     * Position of different data in each row
     *      RawData + QuantizationCodes + Factors + neighborIDs
     * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
     * we do not need to store the degree for each vertex
     */
    size_t code_offset_ = 0;      // pos of packed code
    size_t factor_offset_ = 0;    // pos of Factor
    size_t neighbor_offset_ = 0;  // pos of Neighbors
    // size_t centroid_offset_ = 0;  // pos of Centroid in IVF
    size_t row_offset_ = 0;       // length of entire row
    size_t ef = 0;
    size_t post_ef = 0;
    size_t cluster_id = 0;  // cluster id
    size_t code_size = 0;
    bool allocate_vectors_ = true;  // skip original_data_ allocation to save memory in search-only mode

    void initialize();

    // search on quantized graph
    void search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results, float ep_dist = FLT_MAX
    );

    void copy_vectors(const float*);

    void copy_centroids(const float*);

    void
    find_candidates(PID, size_t, std::vector<Candidate<float>>&, HashBasedBooleanSet&, const std::vector<uint32_t>&)
        const;

    void update_qg(PID, const std::vector<Candidate<float>>&);

    void update_results(buffer::ResultBuffer&, const float*);

    void scan_neighbors(
        const QGQuery& q_obj,
        const PID cur_node,
        float* appro_dist,
        float sqr_y,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree
    ) const;

    // 新版本：使用float LUT的邻居扫描
    void scan_neighbors_float_lut(
        const QGQuery& q_obj,
        const PID cur_node,
        float* appro_dist,
        float sqr_y,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree
    ) const;

   public:
    explicit QuantizedGraph(size_t, size_t, size_t, const FHTRotator::data_type* = nullptr, bool allocate_vectors = true);

    [[nodiscard]] float* get_vector(PID data_id) {
        if (!allocate_vectors_) return nullptr;
        return &original_data_.at(dimension_ * data_id);
    }

    [[nodiscard]] const float* get_vector(PID data_id) const {
        if (!allocate_vectors_) return nullptr;
        return &original_data_.at(dimension_ * data_id);
    }

    /*改为获取某一个 data_id 的 packed_code */
    [[nodiscard]] uint64_t* get_packed_code(PID data_id) {
        return &packed_codes_.at(data_id * code_size);
    }

    [[nodiscard]] const uint64_t* get_packed_code(PID data_id) const {
        return &packed_codes_.at(data_id * code_size);
    }

    [[nodiscard]] float* get_factor(PID data_id) {
        return &factor_codes_.at(data_id * sizeof(Factor) / sizeof(float));
    }
    [[nodiscard]] const FHTRotator::data_type& get_mat() const {
        return this->rotator_.get_mat();
    }

    [[nodiscard]] size_t get_padded_dim() const {
        return padded_dim_;
    }

    [[nodiscard]] const float* get_factor(PID data_id) const {
        return &factor_codes_.at(data_id * sizeof(Factor) / sizeof(float));
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(&data_.at(degree_bound_ * data_id));
    }

    [[nodiscard]] float* get_centroid() {
        return &centroids_.at(0);
    }

    [[nodiscard]] const float* get_centroid() const {
        return &centroids_.at(0);
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at(degree_bound_ * data_id)
        );
    }

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dimension_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save_index(const char*) const;

    void load_index(const char*);

    void set_ef(size_t);
    
    void set_cluster(size_t);

    void set_post_ef(size_t);


    void set_original_data(const float* data);

    void get_lut_sumq(const float* query, float* lut, float* sumq);

    void get_rotated_query(const float* query, float* rotated_query_out, float* sumq_out);

    /* search and copy results to KNN, ep_dist is optional (default FLT_MAX) */
    void search(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results, float ep_dist = FLT_MAX
    );
    // 添加控制计时的方法
    void enable_profiling(bool enable = true) {
        enable_profiling_ = enable;
        if (enable) {
            timer_.reset();
        }
    }
    
    void report_timings() const {
        if (enable_profiling_) {
            timer_.report();
        }
    }
};

inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t dim, const FHTRotator::data_type* mat, bool allocate_vectors)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(dim)
    , padded_dim_(1 << ceil_log2(dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(mat == nullptr ? FHTRotator(dimension_) : FHTRotator(*mat, dimension_))
    , visited_(100)
    , search_pool_(0)
    , allocate_vectors_(allocate_vectors) {
    initialize();
}

inline void QuantizedGraph::copy_vectors(const float* data) {
    if (!allocate_vectors_) return;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const float* src = data + (dimension_ * i);
        float* dst = get_vector(i);
        std::copy(src, src + dimension_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

inline void QuantizedGraph::copy_centroids(const float* c) {
    float* dst = get_centroid();
    std::copy(c, c + dimension_, dst);

}

inline void QuantizedGraph::save_index(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));

    /* Data (neighbors only) */
    data_.save(output);

    /* Packed Codes */
    packed_codes_.save(output);

    /* Factor Codes */
    factor_codes_.save(output);

    /* Centroid */
    centroids_.save(output);

    /* Rotator */
    this->rotator_.save(output);

    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

inline void QuantizedGraph::load_index(const char* filename) {
    // std::cout << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        abort();
    }

    /* Check file size */
    size_t filesize = get_filesize(filename);
    size_t correct_size = sizeof(PID) + 
                          (sizeof(float) * num_points_ * degree_bound_) +
                          (sizeof(uint64_t) * num_points_ * code_size) + 
                          (sizeof(Factor) * num_points_) +
                          (sizeof(float) * dimension_) + 
                          (sizeof(float) * padded_dim_);
    if (filesize != correct_size) {
        std::cerr << "Index file size error! Please make sure the index and "
                     "init parameters are correct\n";
        std::cerr << "Expected size: " << filesize << ", but got: " << correct_size << '\n';
        abort();
    }
    std::cout<< "Loading quantized graph from " << filename << " and checking size: " << filesize << '\n';

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));

    /* Data (neighbors only) */
    data_.load(input);

    /* Packed Codes */
    packed_codes_.load(input);

    /* Factor Codes */
    factor_codes_.load(input);
    
    /* Centroid */
    centroids_.load(input);

    /* Rotator */
    this->rotator_.load(input);

    input.close();
    // std::cout << "Quantized graph loaded!\n";
}
inline void QuantizedGraph::set_post_ef(size_t cur_ef) {
    this->post_ef = cur_ef;
}

inline void QuantizedGraph::set_ef(size_t cur_ef) {
    this->search_pool_.resize(cur_ef);
    this->visited_ = HashBasedBooleanSet(std::min(this->num_points_ / 10, cur_ef * cur_ef));
    visited_packed_.clear();
    this->ef = cur_ef;
}

inline void QuantizedGraph::set_cluster(size_t c_id) {
    cluster_id = c_id;
}

inline void QuantizedGraph::get_lut_sumq(const float* query, float* lut, float* sumq) {
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);
    std::copy(q_obj.float_lut().begin(), q_obj.float_lut().end(), lut);
    std::copy(&q_obj.sumq(), &q_obj.sumq() + 1, sumq);
}

inline void QuantizedGraph::get_rotated_query(
    const float* query, float* rotated_query_out, float* sumq_out) {
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);
    std::copy(
        q_obj.rotated_query_data(),
        q_obj.rotated_query_data() + padded_dim_,
        rotated_query_out
    );
    *sumq_out = q_obj.sumq();
}

/*
 * search single query with custom entry point distance
 */
inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results,
    float ep_dist
) {
    /* Init query matrix */
    this->visited_.clear();
    this->search_pool_.clear();
    search_qg(query, knn, results, ep_dist);
}


inline void QuantizedGraph::set_original_data(const float* data) {
    if (!allocate_vectors_) return;
    float* data_ptr = original_data_.data();
    std::copy(data, data + num_points_ * dimension_, data_ptr);
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   search res
 * @param ep_dist   entry point distance
 */
inline void QuantizedGraph::search_qg(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results, float ep_dist
) {
    if (enable_profiling_) timer_.start("query_preparation");
    // query preparation with float LUT
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);  // 恢复scanner参数，用于pack float LUT
    float sqr_y = 0;
    if (enable_profiling_) timer_.stop();

    if (enable_profiling_) timer_.start("initialization");
    /* Searching pool initialization */
    search_pool_.insert(this->entry_point_, ep_dist);

    /* Result pool */
    buffer::ResultBuffer res_pool(this->post_ef);
    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    // int visited_count = 0;
    if (enable_profiling_) timer_.stop();

    while (search_pool_.has_next()) {
        float cur_dist;
        PID cur_node = search_pool_.pop(cur_dist);
        if (visited_.get(cur_node)) {
            continue;
        }
        visited_.set(cur_node);
        // visited_count++;
        scan_neighbors_float_lut(
            q_obj,
            cur_node,
            appro_dist.data(),
            sqr_y,
            this->search_pool_,
            this->degree_bound_
        );
        if (enable_profiling_) timer_.start("insert_result");
        res_pool.insert(cur_node, cur_dist);
        if (enable_profiling_) timer_.stop();
    }
    // if (enable_profiling_) timer_.start("update_results");
    // update_results(res_pool, query);
    // if (enable_profiling_) timer_.stop();
    // auto res_ids = res_pool.ids();
    // if (enable_profiling_) timer_.start("final_reranking");
    // /* Result pool */
    // buffer::ResultBuffer true_pool(knn);
    // for(size_t i = 0; i < res_pool.size(); i++)
    // {
    //     // float cur_dist = res_dis[i];
    //     PID cur_id = res_ids[i];
    //     float gt_dist = space::l2_sqr(query, get_vector(cur_id), dimension_);
    //     true_pool.insert(cur_id, gt_dist);
    // }

    // true_pool.copy_results(results);
    // if (enable_profiling_) timer_.stop();
    res_pool.copy_results(results, knn);
    // std::cout << "cur_centroid " << get_centroid() << " centroid_size: " << num_points_ << " visited_count: " << visited_count << '\n';
    // // 计算 query 和 res_pool 中的每个点的l2距离, 并和 res_pool 存储的距离比较
    // auto res_dis = res_pool.distances();
    // for (size_t i = 0; i < res_pool.size(); ++i) {
    //     float cur_dist = res_dis[i];
    //     PID cur_id = res_ids[i];
    //     float gt_dist = space::l2_sqr(query, get_vector(cur_id), dimension_);
    //     std::cout << "ef " << this->ef << '\n';
    //     std::cout << "Neighbor ID " << cur_id << '\n';
    //     std::cout << "Appro " << cur_dist << '\t';
    //     std::cout << "GT " << gt_dist << '\t';
    //     std::cout << "Error " << abs(cur_dist - gt_dist) / gt_dist *100 << "%" << '\n';
    // }
}

// 新版本：使用float LUT的邻居扫描
inline void QuantizedGraph::scan_neighbors_float_lut(
    const QGQuery& q_obj,
    const PID cur_node,
    float* appro_dist,
    float sqr_y,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    if (enable_profiling_) timer_.start("scan_init");

    uint64_t* rabit_code = new uint64_t[cur_degree * code_size];
    float* factor = new float[cur_degree * 3];
    const PID* neighbor_ptr = get_neighbors(cur_node);
    if (enable_profiling_) timer_.stop();
    if (enable_profiling_) timer_.start("code_factor_gathering");
    for (size_t i = 0; i < cur_degree; ++i) {
        const uint64_t* packed_code_ptr = get_packed_code(neighbor_ptr[i]);
        // std::cout<< cur_node <<"neighbor_ptr[" << i << "] " << neighbor_ptr[i] << '\n';
        /*存储到rabit_code中*/
        std::memcpy(
            rabit_code + i * code_size, packed_code_ptr, code_size * sizeof(uint64_t)
        );
        const float* factor_ptr = get_factor(neighbor_ptr[i]);
        // std::cout<< cur_node <<"factor_ptr[0] " << factor_ptr[0] << '\n';
        factor[i] = factor_ptr[0];
        // factor[i + this->degree_bound_] = factor_ptr[1];
        // factor[i + this->degree_bound_ * 2] = factor_ptr[2];
    }
    if (enable_profiling_) timer_.stop();
    
    if (enable_profiling_) timer_.start("fastscan");
    // 使用float LUT版本的scanner
    this->scanner_.scan_neighbors_float_lut(
        appro_dist,
        q_obj.float_lut().data(),  // 使用float LUT
        sqr_y,
        q_obj.sumq(),
        rabit_code,
        factor
    );
    if (enable_profiling_) timer_.stop();
    if (enable_profiling_) timer_.start("process_results");
    const PID* ptr_nb = get_neighbors(cur_node);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        // std::cout<< cur_node <<"tmp_dist " << tmp_dist << '\n';

#if defined(DEBUG)
        std::cout << "Neighbor ID " << cur_neighbor << '\n';
        std::cout << "Appro " << appro_dist[i] << '\t';
        float __gt_dist__ = space::l2_sqr(q_obj.query_data(), get_vector(cur_neighbor), dimension_);
        std::cout << "GT " << __gt_dist__ << '\t';
        std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        std::cout << "sqr_y " << sqr_y << '\n';
#endif
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        // memory::mem_prefetch_l2(
        //     reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        // );
    }
    if (enable_profiling_) timer_.stop();
    
    delete[] rabit_code; 
    delete[] factor;
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distnace for current vertex
inline void QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const PID cur_node,
    float* appro_dist,
    float sqr_y,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree
) const {
    if (enable_profiling_) timer_.start("scan_init");


    /* Compute approximate distance by Fast Scan */
    uint64_t* rabit_code = new uint64_t[cur_degree * code_size];
    float* factor = new float[cur_degree * 3];
    const PID* neighbor_ptr = get_neighbors(cur_node);
    if (enable_profiling_) timer_.stop();
    if (enable_profiling_) timer_.start("code_factor_gathering");
    for (size_t i = 0; i < cur_degree; ++i) {
        const uint64_t* packed_code_ptr = get_packed_code(neighbor_ptr[i]);
        /*存储到rabit_code中*/
        std::memcpy(
            rabit_code + i * code_size, packed_code_ptr, code_size * sizeof(uint64_t)
        );
        const float* factor_ptr = get_factor(neighbor_ptr[i]);
        factor[i] = factor_ptr[0];
        factor[i + this->degree_bound_] = factor_ptr[1];
        factor[i + this->degree_bound_ * 2] = factor_ptr[2];
    }
    if (enable_profiling_) timer_.stop();
    // if (enable_profiling_) timer_.start("pack_codes");
    // uint8_t* packed_code = new uint8_t[cur_degree * code_size * 8];
    // pack_codes(padded_dim_, rabit_code, cur_degree, packed_code);
    // if (enable_profiling_) timer_.stop();

    
    if (enable_profiling_) timer_.start("fastscan");
    this->scanner_.scan_neighbors(
        appro_dist,
        q_obj.lut().data(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sumq(),
        rabit_code,
        factor
    );
    if (enable_profiling_) timer_.stop();
    if (enable_profiling_) timer_.start("process_results");
    const PID* ptr_nb = get_neighbors(cur_node);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        // tmp_dist = space::l2_sqr(q_obj.query_data(), get_vector(cur_neighbor), dimension_);
        // if(cluster_id == 1913 && cur_neighbor == 166)
        // {
        //     // 输出 q_obj.query_data() 和 get_vector(cur_neighbor) 的内容
        //     std::cout << "Query vector: ";
        //     for (size_t j = 0; j < dimension_; ++j) {
        //         std::cout << q_obj.query_data()[j] << ' ';
        //     }
        //     std::cout << "\nNeighbor vector: ";
        //     for (size_t j = 0; j < dimension_; ++j) {
        //         std::cout << get_vector(cur_neighbor)[j] << ' ';
        //     }
        //     std::cout << '\n';
        //     std::cout << "Neighbor ID " << cur_neighbor << '\n';
        //     std::cout << "Appro " << appro_dist[i] << '\t';
        //     float __gt_dist__ = space::l2_sqr(q_obj.query_data(), get_vector(cur_neighbor), dimension_);
        //     std::cout << "GT " << __gt_dist__ << '\t';
        //     std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        //     std::cout << "sqr_y " << sqr_y << '\n';
        // }
#if defined(DEBUG)
        std::cout << "Neighbor ID " << cur_neighbor << '\n';
        std::cout << "Appro " << appro_dist[i] << '\t';
        float __gt_dist__ = space::l2_sqr(q_obj.query_data(), get_vector(cur_neighbor), dimension_);
        std::cout << "GT " << __gt_dist__ << '\t';
        std::cout << "Error " << (appro_dist[i] - __gt_dist__) / __gt_dist__ << '\t';
        std::cout << "sqr_y " << sqr_y << '\n';
#endif
        if (search_pool.is_full(tmp_dist) || visited_.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
        // memory::mem_prefetch_l2(
        //     reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        // );
    }
    if (enable_profiling_) timer_.stop();
    
    // delete[] packed_code;
    delete[] rabit_code; 
    delete[] factor;
}

inline void QuantizedGraph::update_results(
    buffer::ResultBuffer& result_pool, const float* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto ids = result_pool.ids();
    for (PID data_id : ids) {
        PID* ptr_nb = get_neighbors(data_id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!visited_.get(cur_neighbor)) {
                visited_.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, space::l2_sqr(query, get_vector(cur_neighbor), dimension_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

inline void QuantizedGraph::initialize() {
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    code_size = padded_dim_ / 64;
    this->code_offset_ = dimension_;  // Pos of packed code (aligned)
    this->factor_offset_ =
        dimension_;  // Pos of Factor
    // this->neighbor_offset_ =
    //     factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
    this->neighbor_offset_ =
        dimension_;
    // this->centroid_offset_ = neighbor_offset_ + degree_bound_;
    this->row_offset_ = neighbor_offset_ + degree_bound_;

    /* Allocate memory of data (neighbors only) */
    data_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 64, false>>(
            std::vector<size_t>{num_points_, degree_bound_}
        );

    /* Allocate memory for original data (vectors)
     * Skip when allocate_vectors_ is false — saves ~89% of per-cluster RAM
     * for search-only (no-exact-reranking) workloads. */
    if (allocate_vectors_) {
        original_data_ = data::
            Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 1 << 22, true>>(
                std::vector<size_t>{num_points_, dimension_}
            );
    }

    /* Allocate memory for packed codes */
    packed_codes_ = data::
        Array<uint64_t, std::vector<size_t>, memory::AlignedAllocator<uint64_t, 128, false>>(
            std::vector<size_t>{num_points_, code_size}
        );

    /* Allocate memory for packed codes */
    factor_codes_ = data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 64, false>>(
            std::vector<size_t>{num_points_ , sizeof(Factor) / sizeof(float)}
        );
    
    /* Allocate memory for centroids */
    centroids_= data::
        Array<float, std::vector<size_t>, memory::AlignedAllocator<float, 64, false>>(
            std::vector<size_t>{dimension_}
        );

    this->visited_packed_ = HashBasedBooleanSet(num_points_);
    visited_packed_.clear();
}

// find candidate neighbors for cur_id, exclude the vertex itself
inline void QuantizedGraph::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<Candidate<float>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const float* query = get_vector(cur_id);
    QGQuery q_obj(query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_);  // 恢复scanner参数，用于pack float LUT
    float sqr_y = space::l2_sqr(q_obj.query_data(), get_centroid(), dimension_);

    /* Searching pool initialization */
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<float> appro_dist(degree_bound_);  // approximate dis
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        scan_neighbors_float_lut(
            q_obj, cur_candi, appro_dist.data(), sqr_y, tmp_pool, cur_degree
        );
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, space::l2_sqr(q_obj.query_data(), get_vector(cur_candi), dimension_));
        }
    }
}

inline void QuantizedGraph::update_qg(
    PID cur_id, const std::vector<Candidate<float>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
    RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
    x_pad.setZero();
    c_pad.setZero();

    /* Copy data */
    for (size_t i = 0; i < cur_degree; ++i) {
        auto neighbor_id = new_neighbors[i].id;
        const auto* cur_data = get_vector(neighbor_id);
        std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<long>(i), 0));
    }
    const auto* cur_cent = get_centroid();
    std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

    /* rotate Matrix */
    RowMatrix<float> x_rotated(cur_degree, padded_dim_);
    RowMatrix<float> c_rotated(1, padded_dim_);
    for (long i = 0; i < static_cast<long>(cur_degree); ++i) {
        this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
    }
    this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    // Get codes and factors for rabitq
    // float* fac_ptr = new float[this->degree_bound_ * 3]; // 不再需要 factor_dq 和 factor_vq
    float* fac_ptr = new float[this->degree_bound_]; // 只需要 triple_x
    float* triple_x = fac_ptr;
    // float* factor_dq = triple_x + this->degree_bound_;
    // float* factor_vq = factor_dq + this->degree_bound_;
    // uint8_t* neighbor_code = new uint8_t[cur_degree * code_size * 8];
    std::vector<uint64_t> rabit_cd(cur_degree * code_size);
    rabitq_codes(
        x_rotated, c_rotated, rabit_cd, triple_x, nullptr, nullptr // factor_dq, factor_vq
    );
    /* Copy neighbor_code 到 packed_codes */
    for (size_t i = 0; i < cur_degree; ++i) {
        uint64_t* packed_code_ptr = get_packed_code(neighbor_ptr[i]);
        /*检查packed_code_ptr 和 neighbor_code 存储的值是否一样，用于debug*/
        if ( visited_packed_.get(neighbor_ptr[i]) &&
                packed_code_ptr[0] != rabit_cd[i * code_size]) {
            std::cout << "Packed code pointer mismatch at index " << i << "in cluster " << cluster_id << '\n';
            std::cout << "Expected: " << rabit_cd[i * code_size] << '\n';
            std::cout << "Got: " << packed_code_ptr[0] << '\n';
            std::cout << "Expected1: " << rabit_cd[i * code_size+1] << '\n';
            std::cout << "Got1: " << packed_code_ptr[1] << '\n';
        }
        visited_packed_.set(neighbor_ptr[i]);
        std::copy(
            rabit_cd.data() + (i * code_size),
            rabit_cd.data() + ((i + 1) * code_size),
            packed_code_ptr
        );
        float* factor_ptr = get_factor(neighbor_ptr[i]);
        factor_ptr[0] = triple_x[i];
        // factor_ptr[1] = factor_dq[i]; // 不再使用
        // factor_ptr[2] = factor_vq[i]; // 不再使用
    }
    delete[] fac_ptr;
    // delete[] neighbor_code;
}
}  // namespace symqg
