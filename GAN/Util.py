import numpy as np
def log(log, mode='debug'):
    print(log)

def one_hot(array, nb_classes):
    targets = np.array(array).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets

def array_to_multi_hot(labels, nb_classes):
    res = np.zeros(nb_classes)
    sums = 0
    for label in labels:
        res[label] = 1
        sums += 1
    if sums > 0:
        res /= sums
    return res