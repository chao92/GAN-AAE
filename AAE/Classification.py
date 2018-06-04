'''
Seperate evaluation of multi_label_classification from implementation of SDNE 

Author: Chao Jiang

'''

#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.io as sio
import sys
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def check_multi_label_classification(X, Y, test_ratio = float(sys.argv[3])):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
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
    print("{}-{}-{}-{}-{}".format(sys.argv[1],sys.argv[2],sys.argv[3],micro,macro))
    #print "micro_f1: %.4f" % (micro)
    #print "macro_f1: %.4f" % (macro)
    
def load_label_data(filename):
    with open(filename,"r") as fin:
        firstLine = fin.readline().strip().split()
        #print firstLine
        label_array = np.zeros([int(firstLine[0]), int(firstLine[1])], np.bool)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split(' : ')
            if len(line) > 1:
                labels = line[1].split()
                for label in labels:
                    label_array[int(line[0])][int(label)] = True
    return label_array

if __name__ == '__main__':
  #embedding_file_name = 'Flickrembedding201_embedding.mat'
  #loadEmbedding = sio.loadmat(embedding_file_name);
  #label_file_name = 'Flickrembedding201_embeding_label.mat'
  #loadEmbeddingLabel = sio.loadmat(label_file_name);
  
  #embedding_file_name = "./Storage/embeddingResult_youtube_0.9686b/"+sys.argv[1]+"-1-0.1-3000-500-0.2-0.2"+sys.argv[2]+"_Link_embedding.mat"
  embedding_file_name = "./embeddingResult/"+sys.argv[1]+"-1000-0.1-3000-500-0.2-0.2"+sys.argv[2]+"_Link_embedding.mat"
  #embedding_file_name = "./"+sys.argv[4]+'/formatted_embeddings/'+sys.argv[1]+'_'+sys.argv[2]+'_embeddings.mat'
  loadEmbedding = sio.loadmat(embedding_file_name);
  label_file_name = './LABLES/'+sys.argv[1]+'_label.mat'
  loadEmbeddingLabel = sio.loadmat(label_file_name);
        
        # print("embedding: ",loadEmbedding['embedding'])
        # print("label: ",loadEmbeddingLabel['label'])
  check_multi_label_classification(loadEmbedding['sortedembedding'], loadEmbeddingLabel[sys.argv[1]+'_label']);

# loadEmbedding = sio.loadmat('./youtube_embedding.mat');
# labels = load_label_data('./youtube-groups.txt')
# print type(embedding) dict
# print(loadEmbedding)
# my_Embedding = Dotdict(loadEmbedding)
# print len(my_Embedding.embedding)
# check_multi_label_classification(my_Embedding.embedding, labels);

#check_link_reconstruction(embedding, graph_data, [10000,30000,50000,70000,90000,100000])
