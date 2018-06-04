import numpy as np
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb
from scipy.sparse import coo_matrix

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)
    
def check_link_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        # print("similarity shape is", similarity.shape)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        # print("data.N is ", data.N)
        # [::-1] reverse order so start from the matrix whose elemet is 1 also indicate there is a edge
        sortedInd = sortedInd[::-1]

        for ind in sortedInd:
            # print("index is ", ind)
            # update here for python 3.5
            x = ind // data.N
            y = ind % data.N
            # print("x and y is", x, y)
            if (x == y):
                continue
            count += 1
            if (data.adj_matrix[x][y] == 1):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))

def check_multi_label_classification(X, Y, test_ratio = 0.8):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)

        sort_index = np.fliplr(np.argsort(y_pred, axis = 1))
        # sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))
    return [micro, macro]
    #############################################

# nc is number of clusters
# to be implemented without the use of any libraries (from the scratch)

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

def check_D_index(embedding, clusters, i):
    embedding=embedding.astype(np.double)
    clf = k_means(embedding, n_clusters=clusters, random_state=i)

    # clf = KMeans(n_clusters=clusters).fit_predict(embedding)

    index_d_val = dunn(clf[1], euclidean_distances(embedding))
    print("The value of Dunn index for a K-Means cluster of size " + str(clusters) + " is: " + str(index_d_val))
    return index_d_val
    
# define KmeanDBI

def K_mean_DBI(embedding, cluster_num):
    clf = KMeans(n_clusters=cluster_num, random_state=1).fit(embedding)
    sigma=np.zeros(cluster_num)
    sigma_sum=np.zeros(cluster_num)
    sigma_count=np.zeros(cluster_num)
    for index in range(0,embedding.shape[0]):
        sigma_count[clf.labels_[index]]+=1
        sigma_sum[clf.labels_[index]]+=np.linalg.norm(clf.cluster_centers_[clf.labels_[index]]-embedding[index])
    sigma=sigma_sum/sigma_count
    dbsum=0
    for label_1 in range(0,cluster_num):
        maxd=0
        for label_2 in range(0,cluster_num):
            if label_2==label_1:
                continue
            r=(sigma[label_1]+sigma[label_2])/np.linalg.norm(clf.cluster_centers_[label_2]-clf.cluster_centers_[label_1])
            if r>maxd:
                maxd=r
        sigma_count[label_1]=maxd
    # print(sigma_count)
    print(np.mean(sigma_count))
    return np.mean(sigma_count)



def K_mean_zhou(original_data, embedding, cluster_num):
    # file_name = "./"+sys.argv[4]+"/formatted_embeddings/"+sys.argv[1]+'_'+sys.argv[2]+'_embeddings.mat'
    # original_data = original_data_sp('./'+sys.argv[4]+'/original_data/'+sys.argv[1]+'.txt.sdne')
    # loadEmbedding = sio.loadmat(file_name)

    clf = KMeans(n_clusters=cluster_num, random_state=1).fit(embedding)
    count=0
    for row in original_data:
        label1=clf.labels_[row[0]]
        label2=clf.labels_[row[1]]
        if(label1==label2):
            count+=1
    print(float(count)/original_data.shape[0])
    return float(count)/original_data.shape[0]




    
