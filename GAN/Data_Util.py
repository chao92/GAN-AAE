from Util import log
from Util import array_to_multi_hot
import random
import numpy as np
from multiprocessing import Pool  

class DataUtil:
    def __init__(self, reindex_path=r'C:\Users\v-sixwu\Downloads\eca_blogCatalog3.txt.labeled.reindex',max_line = -1,test_rate=0, sample_mode=True):
        log('loading Edge-Centic dataset form : %s' % (reindex_path))
        vertex_set = set()
        labels_set = set()
        vertex_list = list()
        label_list = list()
        vertex_label = dict()
        edge_set = set()
        tmp = list()
        
        with open(reindex_path+"_group",'r+',encoding='utf-8') as fin:
            lines = fin.readlines()
            num_class = int(lines[0].strip('\n').split()[1]) - 1
            for line in lines[1:]:
                items = line.strip('\n').split(':')
                vertex = int(items[0].strip())
                labels = [int(x)-1 for x in items[1].strip().split(' ')]
                vertex_label[vertex] = labels
                for label in labels:
                    labels_set.add(label)
            
        with open(reindex_path,'r+',encoding='utf-8') as fin:
            lines = fin.readlines()
            items = lines[1].strip('\n').split()
            self.edge_nums = int(items[1])
            self.vertex_nums = int(items[0])
            for line in lines[1:]:
                items = line.strip('\n').split(' ')
                vertex1 = int(items[0])
                vertex2 = int(items[1])
                edge_set.add((vertex1,vertex2))
                edge_set.add((vertex2,vertex1))
                vertex_set.add(vertex1)
                vertex_set.add(vertex2)
                tmp.append((vertex1,vertex2))

        log('the dataset has been loaded!')
        log('total account of vertex: %d' % len(vertex_set))
        log('total labels of vertex: %d' % len(labels_set))
        log('transforming the dataset')
        n = len(vertex_set)
        self.adj_matrix = np.eye(n,dtype=np.bool)
        for vertex1,vertex2 in tmp:
            self.adj_matrix[vertex1][vertex2] = True
            self.adj_matrix[vertex2][vertex1] = True
            vertex_list.append(vertex1)
            labels1 = vertex_label[vertex1]
            labels2 = vertex_label[vertex2]
            label_list.append(array_to_multi_hot(labels1, num_class))
            vertex_list.append(vertex2)
            label_list.append(array_to_multi_hot(labels2, num_class))

        self.ordered_y = []
        for ox in range(0,len(vertex_set)):
            label = vertex_label[ox]
            self.ordered_y.append(array_to_multi_hot(label, num_class))
        self.ordered_y = np.array(self.ordered_y)


        self.num_class = num_class
        self.num_vertex = len(vertex_set)
        self.x = np.array(vertex_list)
        self.y = np.array(label_list)
        self.ids = range(0, len(vertex_list))
        self.test_num = int(len(self.ids) * test_rate)
        self.train_num = len(self.ids) - self.test_num
        if sample_mode:
            print('Sample Mode')
            self.ids = random.sample(self.ids,len(self.ids))
        self.train_ids = self.ids[0: self.train_num]
        self.test_ids = self.ids[self.train_num:]

        self.infer_step = 0
        self.edge_set = edge_set
        self.edge_list = list(edge_set)

        self.iedge_set = set()

        log('transforming the done!')
        log('train size : %d,  test size: %d' % (self.train_num, self.test_num))

        # 全局的计步器和局部的计步器
        self.global_steps = 0
        self.epcoh_steps = 0
        self.eval_steps = 0

    def set_pointer(self, global_steps, epoch_steps):
        self.global_steps = global_steps
        self.epcoh_steps = epoch_steps

    def generate_negative_set(self,num=10000):
        self.iedge_set = set()
        self.h = []
        self.t = []
        self.ih = []
        self.it = []
        print('Sampling negatives')
        self.train_ids2 = range(num)
        while(len(self.ih) < num):
            x = 0
            y = 0
            while x==y or (x,y) in self.edge_set:
                x = np.random.randint(0, self.num_vertex-1)
                y = np.random.randint(0, self.num_vertex-1)
            self.ih.append(self.adj_matrix[x])
            self.it.append(self.adj_matrix[y])
            i = np.random.randint(0, len(self.edge_list))
            (x,y) = self.edge_list[i]
            self.h.append(self.adj_matrix[x])
            self.t.append(self.adj_matrix[y])
        print('bool -> float32')
        self.h = np.array(self.h)
        self.ih = np.array(self.ih)
        self.t = np.array(self.t)
        self.it = np.array(self.it)
                
        print('Done')

    def random_next_batch(self,batch_size, mode='train'):

        h = []
        t = []
        ih = []
        it = []
        if mode == 'train':
            batch_ids = np.array(random.sample(self.train_ids, batch_size), dtype=np.int32)
            batch_ids2 = np.array(random.sample(self.train_ids2, batch_size), dtype=np.int32)
            h = self.h[batch_ids2]
            t = self.t[batch_ids2]
            ih = self.ih[batch_ids2]
            it = self.it[batch_ids2]

        elif mode == 'test':
            batch_ids = np.array(random.sample(self.test_ids, batch_size), dtype=np.int32)
        x = np.array(self.adj_matrix[self.x[batch_ids],:])
        y = np.array(self.y[batch_ids])
        return x, y,h,t,ih,it

    def next_batch(self, batch_size, mode='train'):

        h = []
        t = []
        ih = []
        it = []
        if mode == 'train':
            start = batch_size * self.epcoh_steps
            end = min(start + batch_size, len(self.train_ids))
            if start >= len(self.train_ids):
                self.epcoh_steps = 0
                raise EOFError('All data have been traversed !')

            self.global_steps += 1
            self.epcoh_steps += 1

            batch_ids = np.array(self.train_ids[start:end], dtype=np.int32)
            batch_ids2 = np.array(random.sample(self.train_ids2, batch_size), dtype=np.int32)
            h = self.h[batch_ids2]
            t = self.t[batch_ids2]
            ih = self.ih[batch_ids2]
            it = self.it[batch_ids2]


        elif mode == 'test':
            # test ids are removed!
            batch_ids = np.array(random.sample(self.train_ids, batch_size), dtype=np.int32)
        x = np.array(self.adj_matrix[self.x[batch_ids],:])
        y = np.array(self.y[batch_ids])
        return x, y,h,t,ih,it


    def next_infer_batch(self,batch_size):
        if self.infer_step < len(self.adj_matrix):
            batch_ids = np.array(range(self.infer_step, min(len(self.adj_matrix),self.infer_step+batch_size)),dtype=np.int32)
            x = np.array(self.adj_matrix[batch_ids])
            y = np.array(self.ordered_y[batch_ids])
            self.infer_step += batch_size
            return x,y
        else:
            raise EOFError()


if __name__ =='__main__':
    test = DataUtil(max_line=100000)
    test.generate_negative_set()
    for i in range(0,5000):
        x, y,h,t,ih,it = test.next_batch(128)
        print(x)
        print(np.shape(x))

