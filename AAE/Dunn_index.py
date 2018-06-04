import numpy as np
import scipy.io as sio
import sys
from sklearn.cluster import k_means
from sklearn.metrics.pairwise import euclidean_distances
import time
# def normalize_to_smallest_integers(labels):
#     """Normalizes a list of integers so that each number is reduced to the minimum possible integer, maintaining the order of elements.
#     :param labels: the list to be normalized
#     :returns: a numpy.array with the values normalized as the minimum integers between 0 and the maximum possible value.
#     """

#     max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
#     sorted_labels = np.sort(np.unique(labels))
#     unique_labels = range(max_v)
#     new_c = np.zeros(len(labels), dtype=np.int32)

#     for i, clust in enumerate(sorted_labels):
#         new_c[labels == clust] = unique_labels[i]

#     return new_c


def dunn(labels, distances):
    """
    Dunn index for cluster validation (the bigger, the better)

    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace

    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, given by the distances between its
    two closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.

    The bigger the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements

    """

    # labels = normalize_to_smallest_integers(labels)

    unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
    max_diameter = max(diameter(labels, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(labels, distances):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    """
    # labels = normalize_to_smallest_integers(labels)
    n_unique_labels = len(np.unique(labels))

    min_distances = np.zeros((n_unique_labels, n_unique_labels))
    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
    return min_distances


def diameter(labels, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :returns:
    """
    # labels = normalize_to_smallest_integers(labels)
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = distances[i, ii]
    return diameters

def check_D_index(embedding, clusters,i):
    clf = k_means(embedding, n_clusters=clusters, random_state=i)

    # clf = KMeans(n_clusters=clusters).fit_predict(embedding)

    index_d_val = dunn(clf[1], euclidean_distances(embedding))
    #print("The value of Dunn index for a K-Means cluster of size " + str(clusters) + " is: " + str(index_d_val))
    return index_d_val

if __name__ == '__main__':
  file_name = "./embeddingResult/"+sys.argv[1]+"-1-0.1-3000-500-2000-0.2"+sys.argv[2]+"_Link_embedding.mat"
  loadEmbedding = sio.loadmat(file_name)
  re=check_D_index(loadEmbedding['sortedembedding'].astype(np.double), int(sys.argv[3]),int(sys.argv[4]))
  print("{}-{}-{}-{}-{}".format(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],re))

# the result of SDNE is:
# The value of Dunn index for a K-Means cluster of size 20 is: 0.876287759801
# The value of Dunn index for a K-Means cluster of size 40 is: 0.842253019952
# The value of Dunn index for a K-Means cluster of size 60 is: 0.815078587556
# The value of Dunn index for a K-Means cluster of size 80 is: 0.800659468174
# The value of Dunn index for a K-Means cluster of size 100 is: 0.793467663113
# time for Dunn index evaluation is  1185.962165594101
