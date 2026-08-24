import math
import random
from dataclasses import dataclass
from typing import List

# ==============================
# Config
# ==============================

@dataclass
class SystemConfig:
    # Measured from single-node experiment (seconds), i.e. all data on 1 node
    base_pim_compute_time: float = 0.00002987   # single-node baseline, per query
    cpu_overhead: float = 0      # per query

    # Network model (InfiniBand 400Gbps)
    alpha: float = 20e-6              # latency (20 microseconds)
    bandwidth_gbps: float = 400       # 400 Gbps

    # ANNS params
    top_k: int = 10
    result_size_bytes: int = 4 * 10   # top-k results: id (4B) each

    # Query vector size for broadcast (SIFT: 128-dim float32)
    query_dim: int = 100
    query_dtype_bytes: int = 4        # float32

    @property
    def query_size_bytes(self) -> int:
        return self.query_dim * self.query_dtype_bytes

    # Merge cost coefficient (measured)
    merge_coeff: float = 1e-7         # tune from experiment

    # Load imbalance factor
    imbalance_std: float = 0.1        # 10% std deviation


# ==============================
# Core Models
# ==============================


def network_time(config: SystemConfig, message_size_bytes: int) -> float:
    """Alpha-Beta model"""
    bandwidth_Bps = config.bandwidth_gbps * 1e9 / 8
    return config.alpha + message_size_bytes / bandwidth_Bps


def simulate_single_query_latency_query_parallel(config: SystemConfig,
                                                  num_nodes: int) -> float:
    """Latency of one query under query parallelism (single assigned node).

    The query is routed to exactly one node that holds the full index.
    No merge is needed; cpu_overhead covers routing/dispatch only.
    Network terms are omitted for N=1 (local execution).
    """
    is_remote = num_nodes > 1
    t_send    = network_time(config, config.query_size_bytes)  if is_remote else 0.0
    noise     = random.gauss(0, config.imbalance_std)
    t_compute = config.base_pim_compute_time * max(0.0, 1 + noise)
    t_collect = network_time(config, config.result_size_bytes) if is_remote else 0.0
    return t_send + t_compute + t_collect + config.cpu_overhead


def simulate_qps_query_parallel(config: SystemConfig, num_nodes: int,
                                 num_queries: int = 3000) -> float:
    """Query parallelism: full dataset replicated on every node.

    Each query is routed to one node (round-robin). N nodes process N
    queries simultaneously, so throughput scales linearly with num_nodes.
    Single-query latency is independent of num_nodes.
    QPS = num_nodes / avg_single_query_latency.
    """
    latencies = [simulate_single_query_latency_query_parallel(config, num_nodes)
                 for _ in range(num_queries)]
    avg_latency = sum(latencies) / len(latencies)
    # N nodes each serve one query in parallel -> Nx throughput
    return num_nodes / avg_latency


# ==============================
# Main Experiment
# ==============================


def run_scalability_experiment():
    config = SystemConfig()

    node_list = [1, 2, 4, 8, 16, 32, 64]

    base_qp = simulate_qps_query_parallel(config, 1)

    header = f"{'Nodes':>6}  {'QueryParallel-QPS':>18}  {'QP-Speedup':>10}"
    print(header)
    print("-" * len(header))

    for n in node_list:
        qp_qps = simulate_qps_query_parallel(config, n)
        print(f"{n:>6}  {qp_qps:>18.2f}  {qp_qps/base_qp:>10.2f}")


if __name__ == "__main__":
    run_scalability_experiment()
