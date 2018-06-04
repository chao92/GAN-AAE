from Multiclassifier_exp import MultiClassificationGAN
from config import Config
from Data_Util import DataUtil

def train_and_test():
    data = DataUtil('/ldev/wsx/tmp/netemb/github/dataset/generated_data/eca_blogCatalog3.txt.labeled.reindex' )
    config = Config()
    config.x_dim = data.num_vertex
    config.input_dim = data.num_vertex
    config.num_class = data.num_class
    config.checkpoint_path = 'netemb_eca_blogCatalog3/'
    gan = MultiClassificationGAN(config)
    gan.init_session()

    for i in range(0,50000):
        X, Y = data.next_batch(config.batch_size)
        res = gan.train_step(X_data=X, Y_data=Y, YS_data=Y)
        if i % 100 == 0:
            if i >0 and i % 2000 == 0:
                gan.save_to_checkpoint()
            samples, labels = gan.figure_step(Y)
            #print(labels)
            X, Y = data.next_batch(config.batch_size,mode='test')
            print(gan.test_step(X_data=X,Y_data=Y))
            print(res)

def infer():
    data = DataUtil('/ldev/wsx/tmp/netemb/github/dataset/generated_data/eca_blogCatalog3.txt.labeled.reindex',sample_mode=False)
    config = Config()
    config.x_dim = data.num_vertex
    config.input_dim = data.num_vertex
    config.num_class = data.num_class
    config.checkpoint_path = 'netemb_eca_blogCatalog3/'
    gan = MultiClassificationGAN(config)
    gan.init_session()

    while True:
        X = data.next_infer_batch(config.batch_size)
        if X is None:
            break;
        res = gan.inference_step(X_data=X)
        print(res)

train_and_test()