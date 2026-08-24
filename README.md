# PIMCQG: Co-Designing Billion-Scale Graph-Based Approximate Nearest Neighbor Search for Processing-in-Memory

**Accepted at SC 2026**

---

## Prerequisites

### Hardware

- PIM Host Machine with UPMEM PIM modules
- Host Memory: 512 GB DDR4 (recommended)
- GPU: NVIDIA A100-SXM4 80GB (optional, for faiss-gpu baseline)

### Software

| Component | Version      |
| --------- | ------------ |
| OS        | Ubuntu 20.04 |
| UPMEM SDK | 2024.2.0     |
| GCC / G++ | 9.4.0        |
| CMake     | 3.31.6+      |
| Python    | 3.10         |

### Installation

**Using Conda (Recommended)**

```bash
# 1. Create and activate conda environment
conda create -n CQG python=3.10 -y
conda activate CQG

# 2. Install Python dependencies
pip install reportlab xlwt matplotlib numpy scipy pandas scikit-learn faiss-cpu

# 3. UPMEM SDK
# Download from: https://drive.google.com/drive/folders/1h4bHc7rhd0T4kJOD9rt391hTykG6EYVJ?usp=sharing
tar -zxf upmem-2024.2.0-Linux-x86_64.tar.gz
source upmem-2024.2.0-Linux-x86_64/upmem_env.sh

# 4. PIMCQG Python bindings (for index building)
cd python/
pip install -r requirements.txt
sh build.sh
cd ..
```

---

## Directory Structure

```
./
├── data/                    # Datasets and indices
├── symqglib/                # Modified SymphonyQG C++ library with PIMCQG index support
│   ├── index/
│   │   ├── fastscan/        # FastScan helper functions
│   │   └── qg/              # Quantized graph
│   ├── third/               # Third-party dependencies
│   └── utils/               # Common utilities
├── python/                  # Python bindings for PIMCQG
├── reproduce/               # Python reproduction code
├── reproduce_cpp/           # C-language PIMCQG implementation (main artifact)
│   ├── src/                 # C source files
│   ├── include/             # Headers
│   ├── dpu/                 # DPU-side code
│   ├── plot_result/         # Figure-generation scripts
│   ├── figure/              # Generated figures (PDF + PNG)
│   ├── run_full.sh          # Unified runner (perf / recall / both)
└── simulator_multi_node_try.py  # Multi-node scalability simulator
```

---

## PIMCQG Artifact (`reproduce_cpp/`)

### Index Building

The artifact supports three billion-scale datasets: [**SIFT1B**](http://corpus-texmex.irisa.fr/), [**SPACEV1B**](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B), and [**SSN1B**](**SSN1B**https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/FB_ssnpp_database.u8bin).

**Pre-built indices are provided** for SIFT1B, SPACE1B, and SSN1B in the `data/` directory. If you already have these, skip to the **Building** section.

If you need to rebuild from scratch:

1. Download the base vectors from the SIFT1B, SPACE1B, SSN1B dataset sources.

2. Download the query sets, and ground truth files from [Google Drive](https://drive.google.com/drive/folders/1CBxwOI-OM787CWQsi4zF6wmxlG0Cccqj?usp=sharing)

3. Cluster the base vectors using `data/ivf.py`. Edit the `DATASETS` list in the script to enable the desired datasets, then run:
```bash
cd data
python ivf.py
```
This produces `{dataset}_centroid_8192.fvecs`, `{dataset}_cluster_id_8192.ivecs`, and `{dataset}_dist_to_centroid_8192.fvecs`.

4. Build the PIMCQG indices:
   - Edit `./reproduce/settings.py` to configure datasets and indexing/querying parameters.
   - Run `python ./reproduce/indexing.py` to build and save indices.

The resulting data layout (see note below about dataset naming):
```
data/
├── sift1B/                              # dataset name must match settings.h
│   ├── sift1B_base.bvecs                # base vectors (recall mode only)
│   ├── sift1B_query.bvecs               # query vectors
│   ├── sift1B_groundtruth.ivecs         # ground truth
│   ├── sift1B_centroid_8192.fvecs       # IVF centroids
│   ├── sift1B_cluster_id_8192.ivecs     # cluster assignments (~8 GB)
│   └── IVF-reduceMem-8192/
│       ├── cluster_to_original_8192.txt
│       └── symphonyqg_32_cluster_{id}.index
├── SPACE1B/
│   ├── vectors.bin/  (directory with vectors_*.bin part files)
│   ├── query.bin / truth.bin
│   └── IVF-reduceMem/  (cluster_to_original_8192.txt + symphonyqg_32_cluster_*.index)
└── SSN/
    ├── FB_ssnpp_database.u8bin
    ├── SSN_queries_subset3000.u8bin / SSN_gt_subset3000.ivecs
    └── IVF-reduceMem/  (cluster_to_original_8192.txt + symphonyqg_32_cluster_*.index)
```

> **Important:** The dataset directory name must match the name in `reproduce_cpp/include/settings.h` (`g_datasets[].name`). By default, lowercase names like `sift1B` are used. If you rename the directory to uppercase (`SIFT1B`), update `settings.h` accordingly.

### Configuration

Before building, set the vector dimensions and datasets under test:

- Edit `./include/settings.h` to configure the datasets and querying parameters.
- In `./run_full.sh`, adjust `DIMM` and `PADDED_DIMM` to match the dataset: SIFT1B (128, 128), SPACE1B (100, 128), SSN (256, 256).

### Building

**Prerequisites:**
```bash
# 1. Source the UPMEM SDK environment
source ../upmem-2024.2.0-Linux-x86_64/upmem_env.sh

# 2. Activate conda environment
conda activate CQG

# 3. Set conda library path (required by run_full.sh at runtime)
export CONDA_LIB_PATH=$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$CONDA_LIB_PATH:$LD_LIBRARY_PATH
```

**Build:**
```bash
cd reproduce_cpp

# Build DPU + host binaries (recommended: handles PADDED_DIMM correctly)
./build.sh

# Build recall binary (pure CPU mode)
make -f Makefile.mixed recall

# Alternatively, build individually:
PADDED_DIMM=128 ./build_dpu.sh              # DPU binary (PADDED_DIMM required!)
make -f Makefile.mixed host                  # bin/host_code
```

| Binary            | Source                  | Mode        |
| ----------------- | ----------------------- | ----------- |
| `bin/host_code`   | `src/main.c`            | Performance |
| `bin/host_recall` | `src/main_cpu_recall.c` | Recall      |

### Running

```bash
cd reproduce_cpp
source ../upmem-2024.2.0-Linux-x86_64/upmem_env.sh

# Performance mode only (DPU-accelerated, measures QPS)
./run_full.sh perf

# Recall mode only (pure CPU, measures recall@K)
./run_full.sh recall

# Both modes sequentially (default)
./run_full.sh
```

**Perf mode** runs PIMCQG via DPU FIFO and reports QPS and latency breakdown.  
**Recall mode** uses the same algorithm logic on CPU with exact L2 distance to verify accuracy.

> **Note on `run_full.sh`:** The script uses `$CONDA_PREFIX/lib` by default for the conda library path. If you are not using conda or your conda environment is at a non-standard location, set `CONDA_LIB_PATH` before running.

### Configurable Parameters

Set as environment variables (passed as compile-time `-D` definitions):

| Parameter              | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `NR_DPUS`              | Physical DPUs allocated                         |
| `VIRTUAL_DPUS`         | Virtual DPU count for load balancing simulation |
| `NR_TASKLETS`          | Tasklets (threads) per DPU                      |
| `BATCH_SIZE`           | Queries per run                                 |
| `DIMM` / `PADDED_DIMM` | Original / padded vector dimension (dataset-specific: see above) |
| `DEGREE`               | Graph node degree for PIMCQG index          |
| `MAX_SIZE_PER_DPU`     | Max data elements per DPU                       |
| `EF`                   | Search candidate set size                       |
| `POST_EF`              | Post-processing candidate set size              |
| `TOPK`                 | Nearest neighbors to return                     |

### Virtual-to-Physical DPU Scaling

The paper's experiments used 2,560 DPUs. For local servers with fewer modules, `NR_DPUS` sets the physical count and `VIRTUAL_DPUS` (2560) enables workload-aware load balancing. The load balancer assigns centroids across virtual DPUs, then maps the top `NR_DPUS` with the highest workloads to physical DPUs. Since CPU-DPU communication is parallelized across ranks and latency is bounded by the busiest DPU, full-scale QPS is extrapolated from that bottleneck. We have validated this extrapolation methodology on the original cloud platform before its decommissioning, confirming that results obtained with fewer physical DPUs closely match the full-scale measurements.

### Batch Processing (QPP)

```bash
# Sweep QUERIES_PER_PUSH values 1–15 with auto-adjusted FIFO depths
./run_qpp_sweep.sh

# Other sweep scripts
./run_nprobe.sh
```

### Output & Visualization

```bash
# Generate all paper figures (PDF + PNG → figure/)
./run_all_plots.sh

# Multi-node scalability simulator
python3 simulator_multi_node_try.py
```

Output files: `nohup_perf_*.out` (perf mode) / `nohup_recall_*.out` (recall mode).

---

## Citation

```bibtex
@inproceedings{pimcqg2026,
  title     = {{PIMCQG}: Co-Designing Billion-Scale Graph-Based Approximate
               Nearest Neighbor Search for Processing-in-Memory},
  author    = {Sitian Chen, Yusen Li, Yao Chen, Minwen Deng, Jintao Meng, Amelie Chi Zhou},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC '26)},
  year      = {2026},
}
```

## License

This project is provided for artifact evaluation and research purposes.