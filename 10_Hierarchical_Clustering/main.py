import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def hierarchical_clustering(data):
    # Initialize clusters with each data point as a cluster
    clusters = [[i] for i in range(len(data))]

    while len(clusters) > 1:
        min_distance = np.inf
        merge_indices = (0, 0)

        # Find the closest clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = np.mean(data[clusters[i]], axis=0)
                cluster_j = np.mean(data[clusters[j]], axis=0)

                distance = euclidean_distance(cluster_i, cluster_j)

                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # Merge the closest clusters
        clusters[merge_indices[0]].extend(clusters[merge_indices[1]])
        del clusters[merge_indices[1]]

    return clusters[0]


# Example usage:
if __name__ == "__main__":
    # Generate some random data for demonstration
    np.random.seed(42)
    data = np.random.rand(10, 2)

    # Apply hierarchical clustering
    result = hierarchical_clustering(data)

    print("Final Cluster:")
    print(result)
