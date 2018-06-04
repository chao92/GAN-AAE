import subprocess
from EdgeGAN import EdgeGAN
from tensorflow.examples.tutorials.mnist import input_data
from config import Config
from Data_Util import DataUtil
import time
import numpy as np
import sys

def main(arg):


    debug = False
    path = arg[0] # '/ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex'
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
    config.batch_size = 256
    config.checkpoint_path += arg[1]+'/'
    gan = EdgeGAN(config,1.0)
    gan.build_graph()
    gan.init_session()


    current_time = time.time()

    with open(arg[2],'w+') as fout:
        print("Predicting start   num_class:%d" % (config.num_class) )
        i = -1
        acc = 0
        total = 0
        while True:
            i += 1
            try:
                X,Y = data.next_infer_batch(config.batch_size)

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
                    #print(probs)
                    lables = np.argmax(probs, axis=-1)
                    return probs, lables

                probs, answers = do_infer(config, X)
                for answer in answers:
                    fout.write('%s\n' % answer)
                truth = []
                for y_line in Y:
                    res = set()
                    for index, y in enumerate(y_line):
                        if y > 0:
                            res.add(index)
                    truth.append(res)
                for index, answer in enumerate(answers):
                    total += 1
                    if answer in truth[index]:
                        acc += 1
            except EOFError as e:
                print(e)
                print("Predicting is finished, total avg : %.4f" % (acc/total))
                if float(acc)/float(total)>0.93:
                    res = subprocess.check_output(["mv", "./model/youtube", "./model/youtube-{}".format(float(acc)/float(total))])
                    res = subprocess.check_output(["mkdir", "./model/youtube"])
                    res = subprocess.check_output(["cp", "./res/eco_youtube_gan.txt", "./res/eco_youtube_gan_{}.txt".format(float(acc)/float(total))])
                break;


if __name__ == "__main__":
    main(sys.argv[1:])
