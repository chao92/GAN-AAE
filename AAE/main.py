import tensorflow as tf
import numpy as np
import scipy.io as sio
from graph import Graph
from config import Config
from aae import AAE
from utils.utils import *
import time
import sys
if __name__ == "__main__":
    arg = sys.argv[1:]
    config = Config(arg)
    config.setClusterNN()
   
    graph_data = Graph(config.file_path, config)
    graph_data.kMeanAdjMatrix_for_AAE_fast(graph_data.adj_matrix, 100, 0, 3)


    graph_data.load_label_data(config.label_file_path)

    mini_batch, mini_batch_index, mini_batch_adjacent_matriX = graph_data.sample()

    adaptive_AutoEncoder = AAE(config, mini_batch_index, mini_batch_adjacent_matriX)

    epochs = 1
    time_consumed = 0

    micro_ave_1 = 0.0
    macro_ave_1 = 0.0

    micro_ave_2 = 0.0
    macro_ave_2 = 0.0

    micro_ave_3 = 0.0
    macro_ave_3 = 0.0

    micro_ave_4 = 0.0
    macro_ave_4 = 0.0

    micro_ave_5 = 0.0
    macro_ave_5 = 0.0

    micro_ave_6 = 0.0
    macro_ave_6 = 0.0

    micro_ave_7 = 0.0
    macro_ave_7 = 0.0

    micro_ave_8 = 0.0
    macro_ave_8 = 0.0

    micro_ave_9 = 0.0
    macro_ave_9 = 0.0

    mini_time = adaptive_AutoEncoder.fit(mini_batch, mini_batch_index, mini_batch_adjacent_matriX)
    time_consumed += mini_time
    while(True):

        mini_batch, mini_batch_index, mini_batch_adjacent_matriX = graph_data.sample()

        mini_time = adaptive_AutoEncoder.fit(mini_batch, mini_batch_index, mini_batch_adjacent_matriX)
        time_consumed += mini_time
        if graph_data.is_epoch_end:
            embedding = None
            loss = 0
            # print("start getting embeding and evaluation")
            while(True):

                # adaptive_AutoEncoder.fetch_mini_batch_from_graph_sample()
                mini_batch, mini_batch_index, mini_batch_adjacent_matriX = graph_data.sample()

                # for i in range(config.clusters):
                #     x_name = str(i)
                #     print("minibatch index of cluster", x_name, "is ", mini_batch_index[x_name])
                step_loss = adaptive_AutoEncoder.get_loss(mini_batch, mini_batch_index, mini_batch_adjacent_matriX)
                loss += step_loss
                # adaptive_AutoEncoder.make_compute_graph()
                if embedding is None:
                    embedding = adaptive_AutoEncoder.get_embedding(mini_batch, mini_batch_index, mini_batch_adjacent_matriX)
                    print("first size of embedding", embedding.shape)
                else:
                    embedding = np.vstack((embedding, adaptive_AutoEncoder.get_embedding(mini_batch, mini_batch_index, mini_batch_adjacent_matriX)))
                    #print("size of embedding", embedding.shape)
                if graph_data.is_epoch_end:
                    break
            print("Epoch : %d Loss : %.3f, Train time consumed : %.3fs"%(epochs, loss, time_consumed))

            if epochs % config.display == 0 and epochs >-1:
            # if epochs % config.display == 0:
                #print("shape of embedding", embedding.shape)
                print(graph_data.label[graph_data.label_index].shape)

                print("------------------------------Sumarry of Epoch %d------------------------------" % (epochs))

                # print("--------------------------Link Prediction Evaluation result--------------------------")

                # sort the embedding file for link prediction
                # print(graph_data.label_index)
                # start_link_prediction = time.time()
                sortedembedding = embedding[graph_data.label_index.argsort()]
                # check_link_reconstruction(sortedembedding, graph_data, [1000, 20000])
                if epochs > 0:
                    sio.savemat(config.embedding_folder + str(epochs)+'_Link_embedding.mat', {'sortedembedding': sortedembedding})

                # print("time for link prediction is ", time.time() - start_link_prediction)

                print("--------------------------Classification Evaluation result--------------------------")
                start_classification = time.time()
                print("evaluation for training 5%")
                [micro_1, macro_1] = check_multi_label_classification(embedding, graph_data.label[graph_data.label_index], test_ratio=0.95)
                micro_ave_1 += micro_1
                macro_ave_1 += macro_1
                print("time for classification is ", time.time() - start_classification)
                if epochs >0:
                    sio.savemat(config.embedding_folder + str(epochs)+'_embedding_label.mat', {'label': graph_data.label[graph_data.label_index]})
                    sio.savemat(config.embedding_folder + str(epochs)+'_embedding.mat', {'embedding': embedding})


                # print("evaluation for training 3%")
                # [micro_2, macro_2] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.97)
                # micro_ave_2 += micro_2
                # macro_ave_2 += macro_2
                # print("evaluation for training 4%")
                # [micro_3, macro_3] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.96)
                # micro_ave_3 += micro_3
                # macro_ave_3 += macro_3
                # print("evaluation for training 5%")
                # [micro_4, macro_4] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.95)
                # micro_ave_4 += micro_4
                # macro_ave_4 += macro_4
                # print("evaluation for training 6%")
                # [micro_5, macro_5] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                    test_ratio=0.94)
                # micro_ave_5 += micro_5
                # macro_ave_5 += macro_5

                # print("evaluation for training 7%")
                # [micro_6, macro_6] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.93)
                # micro_ave_6 += micro_6
                # macro_ave_6 += macro_6

                # print("evaluation for training 8%")
                # [micro_7, macro_7] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.92)
                # micro_ave_7 += micro_7
                # macro_ave_7 += macro_7

                # print("evaluation for training 9%")
                # [micro_8, macro_8] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.91)
                # micro_ave_8 += micro_8
                # macro_ave_8 += macro_8

                # print("evaluation for training 10%")
                # [micro_9, macro_9] = check_multi_label_classification(embedding,
                #                                                       graph_data.label[graph_data.label_index],
                #                                                       test_ratio=0.9)
                # micro_ave_9 += micro_9
                # macro_ave_9 += macro_9

               
                #
                # sio.savemat(config.embedding_folder + str(epochs)+'_embedding.mat', {'embedding': embedding})
                # print("For --------epoches--------- %d micro is %f, macro is %f" % (epochs, micro, macro ))
                # print("--------------------------Link Prediction Evaluation result--------------------------")

                # sort the embedding file for link prediction
                # print(graph_data.label_index)
                # embedding = embedding[graph_data.label_index.argsort()]
                # check_link_reconstruction(embedding, graph_data, [5000, 10000, 20000, 40000, 100000])
                #
                # print("--------------------------DBI Evaluation result--------------------------")
                if epochs==90:
                    print("evaluation for K=50")
                    check_D_index(embedding, 50,1)
                # check_DB_index(embedding, 40)
                # check_DB_index(embedding, 60)
                # check_DB_index(embedding, 80)
                # check_DB_index(embedding, 100)
            epochs += 1

            # k-mean on embedding
            # embedding = embedding[graph_data.label_index.argsort()]
            # graph_data.kMeanAdjMatrix(embedding)

        if epochs > config.epochs_limit:
            print("exceed epochs limit terminating")
            # print("ave_micro_1 is %f, ave_macro_1 is %f" % (micro_ave_1 / config.epochs_limit, macro_ave_1 / config.epochs_limit))
            # print("ave_micro_2 is %f, ave_macro_2 is %f" % (micro_ave_2 / config.epochs_limit, macro_ave_2 / config.epochs_limit))
            # print("ave_micro_3 is %f, ave_macro_3 is %f" % (micro_ave_3 / config.epochs_limit, macro_ave_3 / config.epochs_limit))
            # print("ave_micro_4 is %f, ave_macro_4 is %f" % (micro_ave_4 / config.epochs_limit, macro_ave_4 / config.epochs_limit))
            # print("ave_micro_5 is %f, ave_macro_5 is %f" % (micro_ave_5 / config.epochs_limit, macro_ave_5 / config.epochs_limit))
            # print("ave_micro_6 is %f, ave_macro_6 is %f" % (
            # micro_ave_6 / config.epochs_limit, macro_ave_6 / config.epochs_limit))
            # print("ave_micro_7 is %f, ave_macro_7 is %f" % (
            # micro_ave_7 / config.epochs_limit, macro_ave_7 / config.epochs_limit))
            # print("ave_micro_8 is %f, ave_macro_8 is %f" % (
            # micro_ave_8 / config.epochs_limit, macro_ave_8 / config.epochs_limit))
            # print("ave_micro_9 is %f, ave_macro_9 is %f" % (
            # micro_ave_9 / config.epochs_limit, macro_ave_9 / config.epochs_limit))
            break
