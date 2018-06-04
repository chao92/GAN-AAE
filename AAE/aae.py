import numpy as np
import tensorflow as tf
from config import Config
import itertools
from graph import Graph
import scipy.io as sio
from utils.utils import *
import time

class AAE:
    def __init__(self, config, mini_batch_index, mini_batch_adjacent_matriX):

        self.config = config

        ######### not running out gpu sources ##########
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        ##############---define sparsity variable for each layer---###################
        self.pi_layers = {}
        # initialize pi for each layer
        for i in range(config.clusters):
            name = "encoder_" + str(i)
            self.pi_layers[name] = tf.Variable(tf.random_uniform([config.layer_clusters[i] - 1], 0, 0.1))
            name = "decoder_" + str(i)
            self.pi_layers[name] = tf.Variable(tf.random_uniform([config.layer_clusters[i] - 1], 0, 0.1))

        # --------------------controlling each training point with rho----------------
        self.training_point_rho = tf.Variable(tf.random_uniform([self.config.num_input, 1], 0, 0.5))

        ##############------------define Weight and bias------------##################
        self.W = {}
        self.b = {}

        # initialize W b
        for i in range(config.clusters):
            for j in range(config.layer_clusters[i] - 1):
                name_W = "encoder" + str(i) + "_h" + str(j)
                name_b = "encoder" + str(i) + "_b" + str(j)
                index_struct = "cluster_" + str(i)
                self.W[name_W] = tf.Variable(
                    tf.random_normal([config.structs[index_struct][j], config.structs[index_struct][j + 1]]))
                self.b[name_b] = tf.Variable(tf.random_normal([config.structs[index_struct][j + 1]]))

        for _, value in config.structs.items():
            value = value.reverse()

        for i in range(config.clusters):
            for j in range(config.layer_clusters[i] - 1):
                name_W = "decoder" + str(i) + "_h" + str(j)
                name_b = "decoder" + str(i) + "_b" + str(j)
                index_struct = "cluster_" + str(i)
                self.W[name_W] = tf.Variable(
                    tf.random_normal([config.structs[index_struct][j], config.structs[index_struct][j + 1]]))
                self.b[name_b] = tf.Variable(tf.random_normal([config.structs[index_struct][j + 1]]))

        # reverse agian to be normal
        for _, value in config.structs.items():
            value = value.reverse()

        ##############----define input variable for each cluster----##################
        self.X = {}
        # X is the adj_matrix, X_index and X_size will determined following
        self.X_index = {}
        self.X_size = {}

        # for SDNE loss function
        self.adjacent_matriX = {}

        # initialize X and X_index for each cluster as adj_matrix
        for i in range(config.clusters):
            name = str(i)

            ad_name = "ad_"+str(i)

            self.X[name] = tf.placeholder(tf.float32, [None, config.num_input], name=name)
            print("initialize placeholder X shape is", self.X[name].shape)
            self.adjacent_matriX[name] = tf.placeholder(tf.float32, [None, None], name=ad_name)
            self.X_index[name] = mini_batch_index[name]
            self.X_size[name] = len(mini_batch_index[name])

        ##############################################################################
        self.layer = {}
        self.pred = {}
        self.cluster_center = {}
        self.embedding_layer = {}
        ##############################################################################

        #####################################---Construct computing graph---#########################################
        self.make_compute_graph()
        self.loss = self.__make_loss(self.config)

        # RMSPropOptimizer or AdamOptimizer
        print("constructing optimizer tensor which minimize loss function")
        self.optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("AAE model initialize finish")

    def update_loss_function(self):
        self.loss = self.__make_loss(self.config)

    def fetch_mini_batch_from_graph_sample(self, mini_batch, mini_batch_index, mini_batch_adjacent_matriX):

        input_dict = {}
        for i in range(self.config.clusters):
            x_name = str(i)

            key = tf.get_default_graph().get_operation_by_name(str(i)).outputs[0]
            ad_key = tf.get_default_graph().get_operation_by_name("ad_"+str(i)).outputs[0]

            input_dict[key] = mini_batch[x_name]
            input_dict[ad_key] = mini_batch_adjacent_matriX[x_name]

            # update X_index and X_size
            self.X_index[x_name] = mini_batch_index[x_name]
            self.X_size[x_name] = len(mini_batch_index[x_name])
            # print("length of fit is", self.X_size[x_name])

        return input_dict



    def fetch_All_from_graph_sample(self):

        mini_batch, mini_batch_index, mini_batch_adjacent_matriX = self.graph_data.sampleAll()
        for i in range(self.config.clusters):
            x_name = str(i)
            self.X[x_name] = None
            self.X_index[x_name] = None
            self.adjacent_matriX[x_name] = None

            self.X[x_name] = mini_batch[x_name]
            self.X_index[x_name] = mini_batch_index[x_name]
            self.X_size[x_name] = self.X[x_name].shape[0]

            # for SDNE
            self.adjacent_matriX[x_name] = mini_batch_adjacent_matriX[x_name]
    def get_feed_dict_by_cluster(self, clusterID, mini_batch, mini_batch_index, mini_batch_adjacent_matriX):

        # pass the model.X to placeholder
        input_dict = {}
        x_name = str(clusterID)

        key = tf.get_default_graph().get_operation_by_name(str(clusterID)).outputs[0]
        ad_key = tf.get_default_graph().get_operation_by_name("ad_" + str(clusterID)).outputs[0]

        input_dict[key] = mini_batch[x_name]
        input_dict[ad_key] = mini_batch_adjacent_matriX[x_name]

        return input_dict

    def make_compute_graph(self):
        # Encoder
        def encoder():
            for i in range(self.config.clusters):
                x_name = str(i)
                for j in range(self.config.layer_clusters[i] - 1):
                    layer_name = "encoder_layer" + str(i) + "_" + str(j)
                    name_W = "encoder" + str(i) + "_h" + str(j)
                    name_b = "encoder" + str(i) + "_b" + str(j)
                    if j == 0:
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.X[x_name], self.W[name_W]), self.b[name_b]))
                        if j == self.config.layer_clusters[i] - 2:
                            self.embedding_layer[x_name] = self.layer[layer_name]
                            self.cluster_center[x_name] = tf.reduce_mean(self.layer[layer_name], 0, keep_dims=True)
                    elif j == self.config.layer_clusters[i] - 2:
                        layer_name_pri = "encoder_layer" + str(i) + "_" + str(j - 1)
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.layer[layer_name_pri], self.W[name_W]), self.b[name_b]))
                        self.embedding_layer[x_name] = self.layer[layer_name]
                        self.cluster_center[x_name] = tf.reduce_mean(self.layer[layer_name], 0, keep_dims=True)
                    else:
                        layer_name_pri = "encoder_layer" + str(i) + "_" + str(j - 1)
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.layer[layer_name_pri], self.W[name_W]), self.b[name_b]))

        # Decoder
        def decoder():
            for i in range(self.config.clusters):
                x_name = str(i)
                for j in range(self.config.layer_clusters[i] - 1):
                    layer_name = "decoder_layer" + str(i) + "_" + str(j)
                    name_W = "decoder" + str(i) + "_h" + str(j)
                    name_b = "decoder" + str(i) + "_b" + str(j)
                    if j == 0:
                        layer_name_pri = "encoder_layer" + str(i) + "_" + str(self.config.layer_clusters[i] - 2)
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.layer[layer_name_pri], self.W[name_W]), self.b[name_b]))
                        if j == self.config.layer_clusters[i] - 2:
                            self.pred[x_name] = self.layer[layer_name]
                    elif j == self.config.layer_clusters[i] - 2:
                        layer_name_pri = "decoder_layer" + str(i) + "_" + str(j - 1)
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.layer[layer_name_pri], self.W[name_W]), self.b[name_b]))
                        self.pred[x_name] = self.layer[layer_name]
                    else:
                        layer_name_pri = "decoder_layer" + str(i) + "_" + str(j - 1)
                        self.layer[layer_name] = tf.nn.sigmoid(
                            tf.add(tf.matmul(self.layer[layer_name_pri], self.W[name_W]), self.b[name_b]))
        encoder()
        decoder()
        # print("finishing construct computing graph")

    def clip_prob(self, x):
        return tf.clip_by_value(x, 1e-10, 1-1e-10)
    
    def kl_divergence(self, rho, rho_hat):
        return rho * tf.log(self.clip_prob(rho)) - rho * tf.log(self.clip_prob(rho_hat)) + (1 - rho) * tf.log(self.clip_prob(1 - rho)) - (1 - rho) * tf.log(self.clip_prob(1 - rho_hat))


    def __make_loss(self, config):
        # print("staring computing loss function")

        def get_loss_1(weight):
            loss_1 = 0.0
            for i in range(self.config.clusters):
                x_name = str(i)
                # print("for cluster ", str(i), "self.X is ", self.X[x_name],"self.pred is", self.pred[x_name])
                loss_1 += tf.reduce_sum(tf.pow(self.X[x_name] - self.pred[x_name], 2))
            return weight * loss_1

        def get_loss_2(weight):
            loss_2 = 0.0
            for _, value in self.W.items():
                loss_2 += tf.nn.l2_loss(value)
            return 2 * weight * loss_2

        def get_loss_3(weight):
            loss_3 = 0.0
            for i in range(self.config.clusters):
                x_name = str(i)
                loss_3 += tf.reduce_sum(tf.pow(self.embedding_layer[x_name] - self.cluster_center[x_name], 2))/(
                self.X_size[x_name] * self.X_size[x_name])
            return weight * loss_3

        # combination of n choose 2 centers
        def get_loss_4(weight):
            loss_4 = 0.0
            for combination in list(itertools.combinations(range(self.config.clusters), 2)):
                x_name_1 = str(combination[0])
                x_name_2 = str(combination[1])
                loss_4 += tf.nn.l2_loss(self.cluster_center[x_name_1] - self.cluster_center[x_name_2]) / (
                    self.X_size[x_name_1] * self.X_size[x_name_2])
            return weight * loss_4

        def get_loss_5(weight):
            loss_5 = 0.0
            for i in range(self.config.clusters):
                x_name_encoder = "encoder_" + str(i)
                x_name_decoder = "decoder_" + str(i)
                for j in range(self.config.layer_clusters[i] - 1):
                    layer_name_encoder = "encoder_layer" + str(i) + "_" + str(j)
                    layer_name_decoder = "decoder_layer" + str(i) + "_" + str(j)
                    loss_5 += self.kl_divergence(self.pi_layers[x_name_encoder][j],
                                                 tf.reduce_mean(self.layer[layer_name_encoder]))
                    loss_5 += self.kl_divergence(self.pi_layers[x_name_decoder][j],
                                                 tf.reduce_mean(self.layer[layer_name_decoder]))
            return weight * loss_5

        def map(fn, arrays, dtype=tf.float32):
            # assumes all arrays have same leading dim
            indices = tf.range(tf.shape(arrays[0])[0])
            out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
            return out

        def get_loss_6(weight):
            loss_6 = 0.0
            for i in range(self.config.clusters):
                x_name = str(i)

                for j in range(self.config.layer_clusters[i] - 1):
                    layer_name_encoder = "encoder_layer" + str(i) + "_" + str(j)
                    layer_name_decoder = "decoder_layer" + str(i) + "_" + str(j)

                    index = tf.constant(self.X_index[x_name])
                    selected = tf.gather(self.training_point_rho, index)

                    mapping_f = lambda x,y: self.kl_divergence(x,y)
                    repeated_en = tf.tile(tf.reduce_mean(self.layer[layer_name_encoder], keep_dims=True), [len(self.X_index[x_name]),1])
                    repeated_de = tf.tile(tf.reduce_mean(self.layer[layer_name_decoder], keep_dims=True), [len(self.X_index[x_name]),1])

                    loss_6 += tf.reduce_sum(map(mapping_f, [selected, repeated_en]))
                    loss_6 += tf.reduce_sum(map(mapping_f, [selected, repeated_de]))

            return weight * loss_6

        def get_loss_SDNE_1():
            loss_SDNE_1 = 0.0
            for i in range(self.config.clusters):
                x_name = str(i)
                D = tf.diag(tf.reduce_sum(self.adjacent_matriX[x_name], 1))
                L = D - self.adjacent_matriX[x_name]  ## L is laplation-matriX
                loss_SDNE_1 += 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.embedding_layer[x_name]), L), self.embedding_layer[x_name]))
            return loss_SDNE_1

        def get_loss_SDNE_2(weight):
            loss_SDNE_2 = 0.0
            for i in range(self.config.clusters):
                beta = None
                if i == 0 or i == 1:
                    beta = 20
                else:
                    beta = 10
                x_name = str(i)
                B = self.X[x_name] * (beta - 1) + 1
                loss_SDNE_2 += tf.reduce_sum(tf.pow((self.pred[x_name] - self.X[x_name])*B, 2))
            return weight * loss_SDNE_2

        def get_loss_SDNE_3():
            loss_SDNE_3 = 0.0
            for i in range(self.config.clusters):
                x_name = str(i)
                loss_SDNE_3 += tf.reduce_sum(tf.pow(self.pred[x_name], 2))
            return loss_SDNE_3


        # return 1*get_loss_SDNE_1() + 500*get_loss_SDNE_2()+get_loss_SDNE_3()
        return get_loss_SDNE_2(self.config.relativeWeight[0]) + get_loss_2(self.config.relativeWeight[1]) + get_loss_3(self.config.relativeWeight[2]) - \
               get_loss_4(self.config.relativeWeight[3]) + get_loss_5(self.config.relativeWeight[4]) + get_loss_6(self.config.relativeWeight[5])

    def fit(self, mini_batch, mini_batch_index, mini_batch_adjacent_matriX):
        # return fitting time
        st = time.time()
        self.sess.run(self.optimizer, feed_dict=self.fetch_mini_batch_from_graph_sample(mini_batch, mini_batch_index, mini_batch_adjacent_matriX))
        return time.time() - st

    def get_loss(self, mini_batch, mini_batch_index, mini_batch_adjacent_matriX):

        return self.sess.run(self.loss, feed_dict=self.fetch_mini_batch_from_graph_sample(mini_batch, mini_batch_index, mini_batch_adjacent_matriX))

    def get_embedding(self, mini_batch, mini_batch_index, mini_batch_adjacent_matriX):

        embedding = None
        for i in range(self.config.clusters):
            x_name = str(i)
            # print("embeding cluster", x_name)
            if i == 0:

                embedding = self.sess.run(self.embedding_layer[x_name], feed_dict=self.get_feed_dict_by_cluster(i, mini_batch, mini_batch_index, mini_batch_adjacent_matriX))
                # print("embedding", embedding[0:2,:])

            else:
                # self.make_compute_graph()
                current = self.sess.run(self.embedding_layer[x_name], feed_dict=self.get_feed_dict_by_cluster(i, mini_batch, mini_batch_index, mini_batch_adjacent_matriX))
                # print("embedding", current[0:2, :])
                embedding = np.vstack((embedding, current))
                # print("current embedding shape is", current.shape)
        return embedding

if __name__ == "__main__":
    config = Config()
    config.setClusterNN()
    test = AAE(config)
    # test.train(config.epochs_limit)
    # print(test.W)