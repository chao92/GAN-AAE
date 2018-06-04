class Config(object):
    def __init__(self, arg):

        self.epochs_limit = 92



        # graph data
        
        self.file_path = "./original_data/"+arg[0]+".txt.sdne" #"/ldev/wsx/tmp/netemb/github/dataset/generated_data/vertex_centic/eco_blogCatalog3.txt.sdne"
        self.label_file_path =  "./original_data/"+arg[0]+".txt.sdne_group" #"/ldev/wsx/tmp/netemb/github/dataset/generated_data/vertex_centic/eco_blogCatalog3.txt.sdne_group"
        self.cluster_path = "./original_data/"+arg[0]+".gan.txt"#/ldev/wsx/tmp/netemb/Test-GAN/res.txt"
        
        
        with open(self.label_file_path) as fin:
            line =fin.readline()
            items = line.split(' ')
            print(items)
            vertex_num = int(items[0])
            

        # embedding data
        self.embedding_folder = "embeddingResult/"+arg[0]+"-"+arg[1]+"-"+arg[2]+"-"+arg[3]+"-"+arg[4]+"-"+arg[5]+"-"+arg[6]

        # hyperparameter

        self.learning_rate = 0.001 #0.1-0.001#
        self.display = 10

        # define layer number for each cluster, this also decide the cluster number
        # input layer named layer 1, the total layer number for below definition is [5,7,5].
        # we use the embedding layers below

        self.layer_clusters = [2, 2, 3]
        self.clusters = len(self.layer_clusters)

        # define neural number for each layer within each cluster
        self.num_input = vertex_num
        self.num_embedding = 100

        # pay attention to L4, we subtract it instead of add
        # the loss func is  // w_1 * L1 + w_2 * L2 + w_3 * L3 - w_4 * L4 + w_5 * L5 + w_6 * L6//
        # self.relativeWeight = [1, 0.2, 0.2, 0.2, 0.2, 0.2]
        self.relativeWeight = [float(arg[1]), float(arg[2]), float(arg[3]), float(arg[4]), float(arg[5]), float(arg[6])] #See Google Sheet AAE

    def setClusterNN(self):
        self.structs = {
            'cluster_0': [self.num_input, self.num_embedding],
            'cluster_1': [self.num_input, self.num_embedding],
            'cluster_2': [self.num_input, 500, self.num_embedding],
        }



if __name__ == "__main__":
    print("test config")
    config = Config()
    config.setClusterNN()
    print(config.clusters)
