from graph import Graph
from config import Config
from sklearn.cluster import KMeans
from utils.utils import *
import scipy.io as sio


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = Config()
graph_data = Graph(config.file_path)
graph_data.load_label_data(config.label_file_path)
# graph_data = Graph("GraphData/flickr/flickr.txt")
# graph_data.load_label_data("GraphData/flickr/flickr-groups.txt")
print(graph_data.adj_matrix)
# check_multi_label_classification(graph_data.adj_matrix, graph_data.label)
kmeans = KMeans(n_clusters=3, random_state=0).fit(graph_data.adj_matrix)
# #
cluster_labels = kmeans.labels_.reshape(graph_data.adj_matrix.shape[0], 1)
print(cluster_labels)

for i in range(3):
    cluster_index = np.where(cluster_labels == i)[0]
    print(cluster_index)
# sio.savemat('email-gropus.mat',{'label': kmeans.labels_})
# my_label = np.zeros([1005, 3], np.bool)
# i=0
# print("label")
# for label in kmeans.labels_:
#     print(label)
#     my_label[i][label] = True
#     i += 1
# print(label)

# check_multi_label_classification(graph_data.adj_matrix, my_label)



# loadEmbedding = sio.loadmat('./test_adj_matrix.mat');
#
# my_Embedding = Dotdict(loadEmbedding)
# print("soc adj")
# print(my_Embedding.adj_matrix)
# kmeans = KMeans(n_clusters=3, random_state=0).fit(my_Embedding.adj_matrix)
# # print(kmeans.labels_)
# sio.savemat('soc-gropus.mat',{'label': kmeans.labels_})
# y = np.bincount(kmeans.labels_)
# ii = np.nonzero(y)[0]
# print(np.vstack((ii,y[ii])).T)
#
# my_label = np.zeros([1000, 3], np.bool)
# i=0
# print("label")
# for label in kmeans.labels_:
#     print(label)
#     my_label[i][label] = True
#     i += 1
# # print(label)
#
# check_multi_label_classification(my_Embedding.adj_matrix, my_label)