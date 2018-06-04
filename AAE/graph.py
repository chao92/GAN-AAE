import numpy as np
from config import Config
from sklearn.cluster import KMeans
from utils.utils import *
import random
import copy
import time
import scipy.io as sio

class Graph(object):
    def __init__(self, file_path,config):

        self.config = config

        self.is_epoch_end = False

        # loading data from local disk to initialize the adj_matrix
        fin = open(file_path, "r")
        firstLine = fin.readline().strip().split(" ")
        self.N = int(firstLine[0])
        self.E = int(firstLine[1])
        self.adj_matrix = np.zeros([self.N, self.N], np.float32)
        for line in fin.readlines():
            line = line.strip().split(" ")
            self.adj_matrix[int(line[0]), int(line[1])] += 1
            self.adj_matrix[int(line[1]), int(line[0])] += 1

        fin.close()
        print("getData done")
        print("Vertexs : %d, Edges: %d " % (self.N, self.E))

        # X_index store the training point in the original adj_matrix
        self.X_index = {}

        # X_size stroe the number of each cluster training points
        self.X_size = {}

        # the mini sampling batch size for training
        self.batch_size = {}

        # start index for sampling, initialized when kMean adjMatrix
        self.st = {}

        # after sampling, their corresponding label index
        self.label_index = None

    def kMeanAdjMatrix_for_AAE_fast(self, matrix_data, clusters, random_state, real_clusters=3):
        print("load from mat file for Kmean")
        clusters = [[],[],[]]
        with open(self.config.cluster_path) as fin:
            lines =fin.readlines()
            for i,line in enumerate(lines):
                cluster = int(line.strip('\n'))
                clusters[cluster].append(i)

        self.X_index['0'] = np.array(clusters[0]).reshape(-1,)
        print("shape of x_index is", self.X_index['0'].shape)

        self.X_size['0'] = self.X_index['0'].shape[0]
        print("size of cluster %d is %d" % (0, self.X_size['0']))

        self.X_index['1'] =np.array(clusters[1]).reshape(-1,)
        self.X_size['1'] = self.X_index['1'].shape[0]
        print("size of cluster %d is %d" % (1, self.X_size['1']))

        self.X_index['2'] = np.array(clusters[2]).reshape(-1,)
        self.X_size['2'] = self.X_index['2'].shape[0]
        print("size of cluster %d is %d" % (2, self.X_size['2']))
        print(self.X_index['2'])


        self.batch_size['0'] =  int(self.X_size['0'] / 20)
        self.batch_size['1'] = int(self.X_size['1'] / 20)
        self.batch_size['2'] =  max(10,int(self.X_size['2'] / 20))
        
        key_min = min(self.X_size.keys(), key=(lambda k: self.X_size[k]))
        print("minimal size of cluster is", self.X_size[key_min], "the key is", key_min)

        tmp_iter = self.X_size[key_min] // self.batch_size[key_min]
        for i in range(self.config.clusters):
            name_cluster = str(i)
            if name_cluster != key_min:
                self.batch_size[name_cluster] = self.X_size[name_cluster] // tmp_iter
                print("batch_size of cluster %d is %d" % (i, self.batch_size[name_cluster]))
            self.st[name_cluster] = 0


        """
    def kMeanAdjMatrixDel(self, matrix_data):
        k_cluster = 2
        kmeans = KMeans(n_clusters=k_cluster, random_state = 0).fit(matrix_data)
        cluster_labels = kmeans.labels_.reshape(matrix_data.shape[0], 1)

        size = len(cluster_labels)
        for i in range(k_cluster):
            cluster_index = np.where(cluster_labels == i)[0]
            X = self.adj_matrix[cluster_index, :]
            print("cluster %d with size %d and %f percentage of total"
                  % (i, len(cluster_index), 100.0 * len(cluster_index) / size))
            print("------------density of this cluster is", 1.0 * np.count_nonzero(X) / (len(cluster_index) * size))

        for i in range(self.config.clusters):
            name_cluster = str(i)
            cluster_index = np.where(cluster_labels == i)[0]
            self.X_index[name_cluster] = cluster_index
            self.X_size[name_cluster] = self.X_index[name_cluster].shape[0]

            print("size of cluster %d is %d" % (i, self.X_size[name_cluster]))
            # print("cluster index is ", self.X_index[name_cluster])
            # initialize start index with 0 for each cluster
            self.st[name_cluster] = 0

        # define batch_size for each cluster make sure all node are feeded within bach_iteration times.
        # // you will get divisor only integer value, / you will get divisor can be float value
        # TODO: Be care of the setting of .is_epoch_end.

        key_min = min(self.X_size.keys(), key=(lambda k: self.X_size[k]))
        print("minimal size of cluster is", self.X_size[key_min], "the key is", key_min)

        if self.X_size[key_min] % self.config.batch_iteration == 0:
            self.batch_size[key_min] = self.X_size[key_min] // self.config.batch_iteration
            print("batch_size of cluster %s is %d" % (key_min, self.batch_size[key_min]))
        else:
            self.batch_size[key_min] = self.X_size[key_min] // self.config.batch_iteration + 1
            print("batch_size of cluster %s is %d" % (key_min, self.batch_size[key_min]))

        tmp_iter = self.X_size[key_min] // self.batch_size[key_min]
        for i in range(self.config.clusters):
            name_cluster = str(i)
            if name_cluster != key_min:
                if self.X_size[name_cluster] % tmp_iter == 0:
                    self.batch_size[name_cluster] = self.X_size[name_cluster] // tmp_iter
                else:
                    self.batch_size[name_cluster] = self.X_size[name_cluster] // tmp_iter + 1
                print("batch_size of cluster %d is %d" % (i, self.batch_size[name_cluster]))
"""
 

    def sample(self, do_shuffle=False, with_label = True):
        # print("sample part of graph one time")
        if self.is_epoch_end:
            # restart to store the label index
            self.is_epoch_end = False
            self.label_index = None
            for i in range(self.config.clusters):
                name_cluster = str(i)
                self.st[name_cluster] = 0
            print("epoch end")
            if do_shuffle:
                print("to do with shuffle")
            else:
                print("to do without shuffle")

        mini_batch = {}
        mini_batch_index = {}
        mini_batch_adjacent_matriX = {}


        for i in range(self.config.clusters):
            name_cluster = str(i)

            mini_batch_index[name_cluster] = None
            mini_batch[name_cluster] = None

            # the end point of sampling is decided by the total size of X_size and the batch_size of this cluster
            en = min(self.X_size[name_cluster], self.st[name_cluster] + self.batch_size[name_cluster])
            # print("start from %d to %d"%(self.st[name_cluster], en))
            # print("type of x_index of cluster is", type(self.X_index[name_cluster]))

            mini_batch_index[name_cluster] = self.X_index[name_cluster][self.st[name_cluster]:en]
            # print("index is ", mini_batch_index[name_cluster])

            mini_batch[name_cluster] = self.adj_matrix[mini_batch_index[name_cluster]]

            # for SDNE loss function
            mini_batch_adjacent_matriX[name_cluster] = self.adj_matrix[mini_batch_index[name_cluster]][:,mini_batch_index[name_cluster]]

            # print("batch shape is ", self.mini_batch[name_cluster].shape)
            # append each mini_batch label to this variable for classification evaluation
            if not (self.label_index is None):
                self.label_index = np.hstack((self.label_index, mini_batch_index[name_cluster]))
            else:
                self.label_index = mini_batch_index[name_cluster]
            if(en == self.X_size[name_cluster] and i == self.config.clusters-1):
                en = 0
                self.is_epoch_end = True
            self.st[name_cluster] = en
            # print("sampled batch size is", len(mini_batch_index[name_cluster]))

        return (mini_batch, mini_batch_index, mini_batch_adjacent_matriX)

    def load_label_data(self, filename):
        with open(filename,"r") as fin:
            firstLine = fin.readline().strip().split()
            self.label = np.zeros([self.N, int(firstLine[1])-1], np.bool)
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(' : ')
                if len(line) > 1:
                    labels = line[1].split()
                    for label in labels:
                        self.label[int(line[0])][int(label)-1] = True

if __name__ == '__main__':
    config = Config()

    graph = Graph(config.file_path)
    graph.kMeanAdjMatrix(graph.adj_matrix)

    # for i in range(graph.config.batch_iteration):
    #     print("sample step is ", i)
    #     graph.sample()
    # print("get the label index ", graph.label_index)
    # graph.load_label_data(config.label_file_path)
    # print("get original label shape is ", graph.label.shape)
    # print("original lable top 10", graph.label[0:10,:])
    # print("permuted lable top 10", graph.label[graph.label_index[0:10], :])

