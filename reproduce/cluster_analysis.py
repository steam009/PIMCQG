import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from utils.io import fvecs_read, ivecs_read, read_fvecs, read_ivecs
from settings import datasets
import seaborn as sns


# 计算欧氏距离的平方
def sqr_dist(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2)

# 查找最近的nprobe个质心
def find_nearest_centroids(query, centroids, nprobe):
    """
    找到查询向量最近的nprobe个质心
    
    参数:
        query: 查询向量，形状为(D,)
        centroids: 质心数组，形状为(C, D)
        nprobe: 要返回的最近质心数量
    
    返回:
        list of (distance, centroid_id) 元组，按距离排序
    """
    C = centroids.shape[0]
    centroid_dist = []
    
    # 计算查询向量到每个质心的距离
    for i in range(C):
        dist = sqr_dist(query, centroids[i])
        centroid_dist.append((dist, i))
    
    # 部分排序，只找出最近的nprobe个
    centroid_dist.sort(key=lambda x: x[0])
    return centroid_dist[:nprobe]

def compute_cluster_statistics(dataset, C=32000):
    """
    Compute statistics for each cluster in the dataset.
    """
    # Load data
    base = fvecs_read(f"./data/{dataset}/{dataset}_base.fvecs")
    centroids_path = os.path.join(f'./data/{dataset}/{dataset}_centroid_{C}.fvecs')
    cluster_id_path = os.path.join(f'./data/{dataset}/{dataset}_cluster_id_{C}.ivecs')
    query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
    gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")
    
    centroids = read_fvecs(centroids_path)
    cluster_id = read_ivecs(cluster_id_path)
    
    NQ, D = query.shape
    NQ = 100
    # 记录哪些cluster被选中过以及选中的次数
    cluster_selected = np.zeros(C, dtype=int)
    # 记录每个查询向量的最近质心
    nearest_centroids = [[] for _ in range(NQ)]
    # 对每个查询向量，找到最近的nprobe个质心
    for i in range(NQ):
        # 找到最近的nprobe个质心
        nearest_centroids[i] = find_nearest_centroids(query[i], centroids, 64)
        # 统计每个质心被选中的次数
        for dist, centroid_id in nearest_centroids[i]:
            cluster_selected[centroid_id] += 1
        
    
    N, D = base.shape
    
    # Count points in each cluster
    cluster_counts = np.zeros(C, dtype=int)
    for i in range(C):
        indices = np.where(cluster_id == i)[0]
        cluster_counts[i] = len(indices)
    
    # Compute distances to centroids
    cluster_distances = []
    max_distances = np.zeros(C)
    min_distances = np.ones(C) * float('inf')
    avg_distances = np.zeros(C)
    
    for i in range(C):
        indices = np.where(cluster_id == i)[0]
        
        if len(indices) == 0:
            min_distances[i] = 0
            continue
            
        distances = []
        for idx in indices:
            # Calculate L2 distance between point and its centroid
            dist = np.linalg.norm(base[idx] - centroids[i])
            distances.append(dist)
        
        if distances:
            max_distances[i] = max(distances)
            min_distances[i] = min(distances)
            avg_distances[i] = np.mean(distances)
            cluster_distances.append(distances)
    #只返回cluster_selected不为0的
    cluster_counts = cluster_counts[cluster_selected > 0]
    cluster_distances = [d for i, d in enumerate(cluster_distances) if cluster_selected[i] > 0]
    max_distances = max_distances[cluster_selected > 0]
    min_distances = min_distances[cluster_selected > 0]
    avg_distances = avg_distances[cluster_selected > 0]
    cluster_selected = cluster_selected[cluster_selected > 0]
    return {
        'cluster_counts': cluster_counts,
        'cluster_distances': cluster_distances,
        'max_distances': max_distances,
        'min_distances': min_distances,
        'avg_distances': avg_distances,
        'cluster_selected': cluster_selected
    }

def visualize_cluster_statistics(stats, dataset, C=32000, save_path=None):
    """
    Visualize the statistics of clusters.
    """
    plt.figure(figsize=(18, 12))
    
    # Plot cluster sizes (top 100 largest and smallest for better visualization)
    plt.subplot(2, 3, 1)
    sorted_indices = np.argsort(stats['cluster_counts'])
    smallest_clusters = sorted_indices[:100]
    largest_clusters = sorted_indices[-100:]
    
    plt.bar(np.arange(len(smallest_clusters)), stats['cluster_counts'][smallest_clusters], label="Smallest")
    plt.bar(np.arange(len(smallest_clusters), len(smallest_clusters) + len(largest_clusters)), 
            stats['cluster_counts'][largest_clusters], label="Largest")
    plt.title(f'Cluster Sizes (Smallest 100 and Largest 100)')
    plt.xlabel('Cluster Index (Sorted)')
    plt.ylabel('Number of Points')
    plt.legend()
    
    # Plot histogram of cluster sizes
    plt.subplot(2, 3, 2)
    sns.histplot(stats['cluster_counts'], bins=50, kde=True)
    plt.title(f'Histogram of Cluster Sizes')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    
    # Plot the average distance to centroid
    plt.subplot(2, 3, 3)
    non_zero_avg = stats['avg_distances'][stats['avg_distances'] > 0]
    plt.hist(non_zero_avg, bins=50, alpha=0.7)
    plt.axvline(x=np.mean(non_zero_avg), color='r', linestyle='--', label=f'Mean: {np.mean(non_zero_avg):.4f}')
    plt.axvline(x=np.median(non_zero_avg), color='g', linestyle='--', label=f'Median: {np.median(non_zero_avg):.4f}')
    plt.title(f'Distribution of Average Distances to Centroids')
    plt.xlabel('Average Distance')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot max vs min distances for each non-empty cluster
    plt.subplot(2, 3, 4)
    non_empty_clusters = np.where(stats['cluster_counts'] > 0)[0]
    plt.scatter(stats['min_distances'][non_empty_clusters], 
                stats['max_distances'][non_empty_clusters], 
                alpha=0.5, s=10)
    plt.title('Min vs Max Distance to Centroid for Each Cluster')
    plt.xlabel('Minimum Distance')
    plt.ylabel('Maximum Distance')
    plt.grid(True, alpha=0.3)
    
     # NEW PLOT 1: Cluster selection frequency
    plt.subplot(2, 3, 5)
    # Sort clusters by selection frequency
    sorted_indices = np.argsort(stats['cluster_selected'])
    top_selected = sorted_indices[-100:]  # Top 100 most selected clusters
    
    plt.bar(np.arange(len(top_selected)), stats['cluster_selected'][top_selected])
    plt.title('Top 100 Most Selected Clusters')
    plt.xlabel('Cluster Index (Sorted by Selection Frequency)')
    plt.ylabel('Selection Frequency')
    
    # NEW PLOT 2: Relationship between cluster size and selection frequency
    plt.subplot(2, 3, 6)
    plt.scatter(stats['cluster_counts'], stats['cluster_selected'], alpha=0.5, s=10)
    plt.title('Cluster Size vs Selection Frequency')
    plt.xlabel('Cluster Size (Number of Points)')
    plt.ylabel('Selection Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add a best fit line
    if len(stats['cluster_counts']) > 1:  # Need at least two points for regression
        from scipy import stats as spstats
        slope, intercept, r_value, p_value, std_err = spstats.linregress(
            stats['cluster_counts'], stats['cluster_selected'])
        x = np.linspace(min(stats['cluster_counts']), max(stats['cluster_counts']), 100)
        plt.plot(x, slope*x + intercept, 'r', 
                label=f'Correlation: {r_value:.2f}')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
        
    # Calculate and print summary statistics
    print(f"\nCluster Statistics Summary for {dataset} (C={C}):")
    print(f"Total number of clusters: {C}")
    non_empty_count = np.sum(stats['cluster_counts'] > 0)
    print(f"Number of non-empty clusters: {non_empty_count} ({non_empty_count/C*100:.2f}%)")
    print(f"Number of empty clusters: {C - non_empty_count} ({(C-non_empty_count)/C*100:.2f}%)")
    print(f"Average cluster size: {np.mean(stats['cluster_counts']):.2f}")
    print(f"Median cluster size: {np.median(stats['cluster_counts']):.2f}")
    print(f"Largest cluster: {np.max(stats['cluster_counts'])} points")
    print(f"Smallest non-empty cluster: {np.min(stats['cluster_counts'][stats['cluster_counts'] > 0])} points")

if __name__ == "__main__":
    # Create output directory for visualizations
    os.makedirs("./cluster_analysis", exist_ok=True)
    
    for DATASET in datasets.keys():
        print(f"\nAnalyzing dataset: {DATASET}")
        C = 4096  # Number of clusters
        
        # Compute statistics
        stats = compute_cluster_statistics(DATASET, C)
        
        # Visualize statistics
        save_path = f"./cluster_analysis/{DATASET}_cluster_statistics_IVF{C}.png"
        visualize_cluster_statistics(stats, DATASET, C, save_path)
        
        # Save the statistics
        stats_path = f"./cluster_analysis/{DATASET}_cluster_statistics_IVF{C}.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Saved statistics to {stats_path}")