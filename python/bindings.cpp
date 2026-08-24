#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"

namespace py = pybind11;
using py_float_array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_uint_array = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

namespace {
void get_arr_shape(const py::buffer_info& buffer, size_t& rows, size_t& cols) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        std::cerr << "Input data has an incorrect shape. Data must be a 1D or 2D array.\n";
        return;
    }
    if (buffer.ndim == 2) {
        rows = buffer.shape[0];
        cols = buffer.shape[1];
    } else {
        rows = 1;
        cols = buffer.shape[0];
    }
}

// 将 numpy 数组转换为 Array 类型
symqg::FHTRotator::data_type numpy_to_array(const py::array_t<float>& np_array) {
    auto buf = np_array.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }
    
    size_t size = buf.shape[0];
    
    // 创建 Array 对象
    symqg::FHTRotator::data_type arr(std::vector<size_t>{size});
    
    // 复制数据
    float* data_ptr = arr.data();
    const float* src_ptr = static_cast<const float*>(buf.ptr);
    std::copy(src_ptr, src_ptr + size, data_ptr);
    
    return arr;
}

// 将 Array 类型转换为 numpy 数组
py::array_t<float> array_to_numpy(const symqg::FHTRotator::data_type& arr, size_t size) {
    const float* data_ptr = arr.data();
    
    // 创建一个拷贝的 numpy 数组，因为我们无法保证原始数据的生命周期
    auto result = py::array_t<float>(size);
    auto buf = result.request();
    float* result_ptr = static_cast<float*>(buf.ptr);
    std::copy(data_ptr, data_ptr + size, result_ptr);
    
    return result;
}
}  // namespace

struct Index {
    std::unique_ptr<symqg::QuantizedGraph> index = nullptr;

    explicit Index(
        const std::string& index_type,
        const std::string& metric,
        size_t num_points,
        size_t dim,
        size_t degree
    ) {
        if (metric != "L2") {
            std::cerr << "Only L2 distance supported currently\n";
            return;
        }

        if (degree < 32 || degree % 32 != 0) {
            std::cerr << "The degree bound must be a multiple of 32\n";
            return;
        }

        if (index_type == "QG") {
            index = std::make_unique<symqg::QuantizedGraph>(num_points, degree, dim, nullptr);
        } else {
            std::cerr << "Index type [" << index_type << "] not supported\n";
            return;
        }
    }

    explicit Index(
        const std::string& index_type,
        const std::string& metric,
        size_t num_points,
        size_t dim,
        size_t degree,
        const py::array_t<float>& mat_array
    ) {
        if (metric != "L2") {
            std::cerr << "Only L2 distance supported currently\n";
            return;
        }

        if (degree < 32 || degree % 32 != 0) {
            std::cerr << "The degree bound must be a multiple of 32\n";
            return;
        }

        if (index_type == "QG") {
            // 将 numpy 数组转换为 Array 类型并创建索引
            // 我们需要先创建 Array，然后将其传递给 QuantizedGraph
            auto mat = std::make_unique<symqg::FHTRotator::data_type>(numpy_to_array(mat_array));
            index = std::make_unique<symqg::QuantizedGraph>(num_points, degree, dim, mat.get());
        } else {
            std::cerr << "Index type [" << index_type << "] not supported\n";
            return;
        }
    }

    void load(const std::string& filename) const { index->load_index(filename.c_str()); }

    void save(const std::string& filename) const { index->save_index(filename.c_str()); }


    void set_original_data(py_float_array& data) const {
        index->set_original_data(data.data(0));
    }

    void set_ef(size_t ef_search) const { index->set_ef(ef_search); }
    void set_cluster(size_t c_id) const { index->set_cluster(c_id); }
    void set_post_ef(size_t ef_search) const { index->set_post_ef(ef_search); }
    void enable_profiling(bool enable) {
        index->enable_profiling(enable);
    } 
    void report_timings() const {
        index->report_timings();
    }
    py::array_t<float> get_mat() const {
        return array_to_numpy(index->get_mat(), get_padded_dim());
    }

    size_t get_padded_dim() const {
        return index->get_padded_dim();
    }
    void build_index(
        const py::object& data,
        size_t ef_indexing,
        size_t num_iter = 3,
        size_t num_threads = UINT_MAX,
        const py::object& centroids = py::none()
    ) const {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
        auto buffer = items.request();
        size_t num = 0;
        size_t dim = 0;
        get_arr_shape(buffer, num, dim);
        //转换centroids vector
        py::array_t<float, py::array::c_style | py::array::forcecast> centroids_arr(centroids);
        if (num != index->num_vertices() || dim != index->dimension()) {
            std::cerr
                << "The shape of data is different with initialization! Expected shape: ("
                << index->num_vertices() << ", " << index->dimension() << "), but got: ("
                << num << ", " << dim << ")\n";
            return;
        }
        symqg::QGBuilder builder(*index, ef_indexing, items.data(), num_threads, centroids_arr.data());
        builder.build(num_iter);
        std::cout << "\tQuantizedGraph created\n";
    }

    auto search(py_float_array& query, uint32_t knn) const {
        py_uint_array result(knn);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);
        // 初始化为 UINT32_MAX，表示无效结果
        std::fill(result_ptr, result_ptr + knn, UINT32_MAX);
        index->search(query.data(0), knn, result_ptr);

        return result;
    }

    auto search_with_ep_dist(py_float_array& query, uint32_t knn, float ep_dist) const {
        py_uint_array result(knn);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);
        // 初始化为 UINT32_MAX，表示无效结果
        std::fill(result_ptr, result_ptr + knn, UINT32_MAX);
        index->search(query.data(0), knn, result_ptr, ep_dist);

        return result;
    }
    auto get_entry_point() const {
        return index->entry_point();
    }
};

PYBIND11_MODULE(symphonyqg, m) {
    m.doc() = R"pbdoc(Towards Symphonious Integration of Graph and Quantization)pbdoc";

    py::class_<Index>(m, "Index")
        .def(
            py::init<const std::string&, const std::string&, size_t, size_t, size_t>(),
            py::arg("index_type"),
            py::arg("metric"),
            py::arg("num_elements"),
            py::arg("dimension"),
            py::arg("degree_bound") = 32
        )
        .def(
            py::init<const std::string&, const std::string&, size_t, size_t, size_t, const py::array_t<float>&>(),
            py::arg("index_type"),
            py::arg("metric"),
            py::arg("num_elements"),
            py::arg("dimension"),
            py::arg("degree_bound"),
            py::arg("mat")
        )
        .def("load", &Index::load, py::arg("filename"))
        .def("save", &Index::save, py::arg("filename"))
        .def("set_original_data", &Index::set_original_data, py::arg("data"))
        .def("set_ef", &Index::set_ef, py::arg("EF"))
        .def("set_cluster", &Index::set_cluster, py::arg("c_id"))
        .def("set_post_ef", &Index::set_post_ef, py::arg("EF"))
        .def("enable_profiling", &Index::enable_profiling, py::arg("enable"))
        .def("report_timings", &Index::report_timings)
        .def("get_mat", &Index::get_mat)
        .def(
            "build_index",
            &Index::build_index,
            py::arg("data"),
            py::arg("EF"),
            py::arg("num_iter") = 3,
            py::arg("num_thread") = UINT_MAX,
            py::arg("centroids")
        )
        .def("search", &Index::search, py::arg("query"), py::arg("k"))
        .def("search", &Index::search_with_ep_dist, py::arg("query"), py::arg("k"), py::arg("ep_dist"))
        .def("get_entry_point", &Index::get_entry_point);
}
