import numpy as np
from sklearn.cluster import k_means
from scipy.spatial import distance
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb

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

def compute_s(i, x, labels, clusters):
	norm_c= len(clusters)
	s = 0
	for x in clusters:
		# print x
		s += distance.euclidean(x, clusters[i])
	return s

def compute_Rij(i, j, x, labels, clusters, nc):
	Rij = 0
	try:
		# print "h"
		d = distance.euclidean(clusters[i],clusters[j])
		# print d
		Rij = (compute_s(i, x, labels, clusters) + compute_s(j, x, labels, clusters))/d
		# print Rij
	except:
		Rij = 0
	return Rij

def compute_R(i, x, labels, clusters, nc):
	list_r = []
	for i in range(nc):
		for j in range(nc):
			if(i!=j):
				temp = compute_Rij(i, j, x, labels, clusters, nc)
				list_r.append(temp)

	return max(list_r)

def compute_DB_index(x, labels, clusters, nc):
	# print x
	sigma_R = 0.0
	for i in range(nc):
		sigma_R = sigma_R + compute_R(i, x, labels, clusters, nc)

	DB_index = float(sigma_R)/float(nc)
	return DB_index

def check_DB_index(embedding, clusters):
    clf = k_means(embedding, n_clusters=clusters)
    centroids = clf[0]
    labels = clf[1]
    index_db_val = compute_DB_index(embedding, labels, centroids, clusters)
    print("The value of Davies Bouldin index for a K-Means cluser of size " + str(clusters) + " is: " + str(index_db_val))
    return index_db_val

    
