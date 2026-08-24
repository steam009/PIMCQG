#include "../include/symqg_c_binding.h"
#include "../../symqglib/qg/qg.hpp"
#include "../../symqglib/utils/buffer.hpp"
#include <iostream>
#include <exception>

/* C++ to C translation layer */

extern "C" {

/* Use C++ objects as opaque pointers */
struct QuantizedGraph_C {
    symqg::QuantizedGraph* cpp_obj;
};

struct ResultBuffer_C {
    symqg::buffer::ResultBuffer* cpp_obj;
    size_t capacity;  // save capacity info for reconstruction on clear
};

QuantizedGraph_C* qg_create(size_t num_elements,
                            size_t degree_bound,
                            size_t dimension,
                            int allocate_vectors) {
    try {
        QuantizedGraph_C* wrapper = new QuantizedGraph_C();
        wrapper->cpp_obj = new symqg::QuantizedGraph(
            num_elements,
            degree_bound,
            dimension,
            nullptr,  // rotation matrix (use default)
            allocate_vectors != 0  // convert int to bool
        );
        return wrapper;
    } catch (const std::exception& e) {
        std::cerr << "Error creating QuantizedGraph: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error creating QuantizedGraph" << std::endl;
        return nullptr;
    }
}

void qg_free(QuantizedGraph_C* qg) {
    if (qg) {
        if (qg->cpp_obj) {
            delete qg->cpp_obj;
        }
        delete qg;
    }
}

int qg_load_index(QuantizedGraph_C* qg, const char* filename) {
    if (!qg || !qg->cpp_obj || !filename) {
        return -1;
    }
    
    try {
        qg->cpp_obj->load_index(filename);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error loading index" << std::endl;
        return -1;
    }
}

int qg_save_index(QuantizedGraph_C* qg, const char* filename) {
    if (!qg || !qg->cpp_obj || !filename) {
        return -1;
    }
    
    try {
        qg->cpp_obj->save_index(filename);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error saving index: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error saving index" << std::endl;
        return -1;
    }
}

void qg_set_ef(QuantizedGraph_C* qg, size_t ef_search) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->set_ef(ef_search);
    }
}

void qg_set_cluster(QuantizedGraph_C* qg, size_t c_id) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->set_cluster(c_id);
    }
}

void qg_set_post_ef(QuantizedGraph_C* qg, size_t ef_search) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->set_post_ef(ef_search);
    }
}

void qg_enable_profiling(QuantizedGraph_C* qg, int enable) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->enable_profiling(enable != 0);
    }
}

void qg_report_timings(QuantizedGraph_C* qg) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->report_timings();
    }
}

int qg_search(QuantizedGraph_C* qg, 
              const float* query, 
              uint32_t knn, 
              uint32_t* results,
              float ep_dist) {
    if (!qg || !qg->cpp_obj || !query || !results) {
        return -1;
    }
    
    try {
        qg->cpp_obj->search(query, knn, results, ep_dist);
        return static_cast<int>(knn);
    } catch (const std::exception& e) {
        std::cerr << "Error during search: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error during search" << std::endl;
        return -1;
    }
}

size_t qg_get_num_elements(const QuantizedGraph_C* qg) {
    if (!qg || !qg->cpp_obj) {
        return 0;
    }
    return qg->cpp_obj->num_vertices(); 
}

size_t qg_get_dimension(const QuantizedGraph_C* qg) {
    if (!qg || !qg->cpp_obj) {
        return 0;
    }
    /* Return dimension */
    return qg->cpp_obj->dimension();
}

void qg_get_lut_sumq(QuantizedGraph_C* qg, const float* query, float* lut, float* sumq) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->get_lut_sumq(query, lut, sumq);
    }
}

void qg_get_rotated_query(QuantizedGraph_C* qg, const float* query,
                          float* rotated_query_out, float* sumq_out) {
    if (qg && qg->cpp_obj) {
        qg->cpp_obj->get_rotated_query(query, rotated_query_out, sumq_out);
    }
}

const float* qg_get_vector(const QuantizedGraph_C* qg, uint32_t data_id) {
    if (!qg || !qg->cpp_obj) {
        return nullptr;
    }
    
    try {
        return qg->cpp_obj->get_vector(data_id);
    } catch (const std::exception& e) {
        std::cerr << "Error getting vector: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error getting vector" << std::endl;
        return nullptr;
    }
}

const uint64_t* qg_get_packed_code(const QuantizedGraph_C* qg, uint32_t data_id) {
    if (!qg || !qg->cpp_obj) {
        return nullptr;
    }
    
    try {
        return qg->cpp_obj->get_packed_code(data_id);
    } catch (const std::exception& e) {
        std::cerr << "Error getting packed code: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error getting packed code" << std::endl;
        return nullptr;
    }
}

const float* qg_get_factor(const QuantizedGraph_C* qg, uint32_t data_id) {
    if (!qg || !qg->cpp_obj) {
        return nullptr;
    }
    
    try {
        return qg->cpp_obj->get_factor(data_id);
    } catch (const std::exception& e) {
        std::cerr << "Error getting factor: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error getting factor" << std::endl;
        return nullptr;
    }
}

const uint32_t* qg_get_neighbors(const QuantizedGraph_C* qg, uint32_t data_id) {
    if (!qg || !qg->cpp_obj) {
        return nullptr;
    }
    
    try {
        return reinterpret_cast<const uint32_t*>(qg->cpp_obj->get_neighbors(data_id));
    } catch (const std::exception& e) {
        std::cerr << "Error getting neighbors: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error getting neighbors" << std::endl;
        return nullptr;
    }
}

uint32_t qg_get_entry_point(const QuantizedGraph_C* qg) {
    if (!qg || !qg->cpp_obj) {
        return 0;
    }
    
    try {
        return qg->cpp_obj->entry_point();
    } catch (const std::exception& e) {
        std::cerr << "Error getting entry point: " << e.what() << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "Unknown error getting entry point" << std::endl;
        return 0;
    }
}

/* =============== ResultBuffer C Binding Implementation =============== */

ResultBuffer_C* result_buffer_create(size_t capacity) {
    try {
        ResultBuffer_C* wrapper = new ResultBuffer_C();
        wrapper->cpp_obj = new symqg::buffer::ResultBuffer(capacity);
        wrapper->capacity = capacity;
        return wrapper;
    } catch (const std::exception& e) {
        std::cerr << "Error creating ResultBuffer: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Unknown error creating ResultBuffer" << std::endl;
        return nullptr;
    }
}

void result_buffer_free(ResultBuffer_C* buffer) {
    if (buffer) {
        if (buffer->cpp_obj) {
            delete buffer->cpp_obj;
        }
        delete buffer;
    }
}

void result_buffer_insert(ResultBuffer_C* buffer, uint32_t data_id, float distance) {
    if (buffer && buffer->cpp_obj) {
        buffer->cpp_obj->insert(data_id, distance);
    }
}

int result_buffer_is_full(const ResultBuffer_C* buffer) {
    if (!buffer || !buffer->cpp_obj) {
        return 0;
    }
    return buffer->cpp_obj->is_full() ? 1 : 0;
}

size_t result_buffer_size(const ResultBuffer_C* buffer) {
    if (!buffer || !buffer->cpp_obj) {
        return 0;
    }
    return buffer->cpp_obj->size();
}

size_t result_buffer_copy_results(const ResultBuffer_C* buffer, 
                                  uint32_t* ids, 
                                  float* distances, 
                                  size_t max_size) {
    if (!buffer || !buffer->cpp_obj || !ids) {
        return 0;
    }
    
    try {
        const auto& result_ids = buffer->cpp_obj->ids();
        const auto& result_distances = buffer->cpp_obj->distances();
        size_t copy_size = std::min(max_size, buffer->cpp_obj->size());
        
        // Copy IDs
        for (size_t i = 0; i < copy_size; ++i) {
            ids[i] = result_ids[i];
        }
        
        // Copy distances (if distance array provided)
        if (distances) {
            for (size_t i = 0; i < copy_size; ++i) {
                distances[i] = result_distances[i];
            }
        }
        
        return copy_size;
    } catch (const std::exception& e) {
        std::cerr << "Error copying results: " << e.what() << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "Unknown error copying results" << std::endl;
        return 0;
    }
}

void result_buffer_clear(ResultBuffer_C* buffer) {
    if (buffer && buffer->cpp_obj) {
        // ResultBuffer has no direct clear method; we need to recreate the object
        delete buffer->cpp_obj;
        buffer->cpp_obj = new symqg::buffer::ResultBuffer(buffer->capacity);
    }
}

void result_buffer_merge(ResultBuffer_C* dest, ResultBuffer_C* src) {
    if (!dest || !dest->cpp_obj || !src || !src->cpp_obj) {
        return;
    }
    
    try {
        // Get all results from source buffer
        const auto& src_ids = src->cpp_obj->ids();
        const auto& src_distances = src->cpp_obj->distances();
        size_t src_size = src->cpp_obj->size();
        
        // Insert each element from source buffer into destination buffer
        for (size_t i = 0; i < src_size; ++i) {
            dest->cpp_obj->insert(src_ids[i], src_distances[i]);
        }
        // Clear source buffer
        result_buffer_clear(src);
    } catch (const std::exception& e) {
        std::cerr << "Error merging result buffers: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error merging result buffers" << std::endl;
    }
}

} // extern "C"

