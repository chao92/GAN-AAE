from EdgeGAN import EdgeGAN
from tensorflow.examples.tutorials.mnist import input_data
from config import Config
from Data_Util import DataUtil
import time
import numpy as np
import sys
import os







def main(argv):

    # /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex BlogTest
    debug = False
    path = argv[0] #'/ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex'
    config = Config()
    if not debug:
        print('normal mode')
        data = DataUtil(path)
        config.x_dim = data.num_vertex
        config.input_dim = data.num_vertex
        config.num_class = data.num_class
        config.batch_size = 16
        data.generate_negative_set()
    else:
        path = r'C:\Users\v-sixwu\Downloads\all.txt'
        data = DataUtil(path)
        config.x_dim = data.num_vertex
        config.input_dim = data.num_vertex
        config.num_class = data.num_class
        config.batch_size = 16
        data.generate_negative_set(1000)

    config.checkpoint_path += argv[1]+'/'
    config.max_step = int(data.edge_nums * config.train_ratio / config.batch_size) + 1
    config.max_step = max(config.max_step, 5005)
    config.max_step = min(config.max_step, 15005)
    print("set max step to: %d" % config.max_step)
    lr = float(argv[2])
    gan = EdgeGAN(config,lr=lr)
    gan.build_graph()
    gan.init_session()



    current_time = time.time()


    for epoch in range(config.epochs):
        print("Epoch %d start" % epoch)
        i = -1
        while True:
            i += 1
            if i >= config.max_step:
                gan.save_to_checkpoint()
                break
            try:
                X, Y, h, t, ih, it = data.random_next_batch(config.batch_size, 'train')
            except EOFError as e:
                print(e)
                print("Epoch %d is finished" % epoch)
                break;
            res = gan.train_step(X, Y, h, t, ih, it)
            if i % config.show_res_per_steps == 0:
                print(res)
            if i % config.checkpoint_per_steps == 0:
                gan.save_to_checkpoint()
            if i % config.negative_sampling_per_step == 0:
                    negative_tuple_nums = config.batch_size * config.negative_sampling_per_step
                    data.generate_negative_set(negative_tuple_nums)
            if i % config.internal_test_per_steps == 0:
                    X, Y, h, t, ih, it = data.next_batch(config.batch_size*2, 'test')
                    X, Y = data.next_infer_batch(config.batch_size)
                    def do_infer(config, X_data):
                        num_class = config.num_class
                        MX = []
                        MY = []
                        for x in X_data:
                            for i in range(num_class):
                                MX.append(x)
                                y = np.zeros([num_class])
                                y[i] = 1.0
                                MY.append(y)

                        probs = gan.infer_step(MX, MY)
                        probs = np.reshape(probs, [-1, num_class])
                        print(probs)
                        lables = np.argmax(probs, axis=-1)
                        return probs, lables

                    probs,answers = do_infer(config,X)
                    print("Testing:#########")
                    base_scores = 0
                    truth = []
                    for y_line in Y:
                        res = set()
                        for index, y in enumerate(y_line):
                            if y > 0:
                                res.add(index)
                                base_scores += 1
                        truth.append(res)
                    base_scores /= (config.num_class * len(Y))


                    acc = 0
                    for index, answer in enumerate(answers):
                        if answer in truth[index]:
                            acc += 1
                    print(np.argmax(Y, axis=-1))
                    print(answers)
                    print("%f\t%f" % (base_scores, acc/len(Y)))
                    print("Testing:#########")

if __name__ == "__main__":
    main(sys.argv[1:])
