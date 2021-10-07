import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import argparse


if __name__ == '__main__':


    PATH_t = './batchsize=128_maxepochs=100/Statistics/Training_accuracy.pickle'
    PATH_v = './batchsize=128_maxepochs=100/Statistics/Validation_accuracy.pickle'
    with open(PATH_t, 'rb') as file:
        tl = pickle.load(file)

    with open(PATH_v, 'rb') as file:
        vl = pickle.load(file)


    mtl = []
    mvl = []

    print(len(vl))
    for s in tl:
        print('\t', len(s))
        for l in s:
            print('\t \t', len(l))
            
            mtl.append(np.mean(l))


    for s in vl:
        for l in s:
            mvl.append(np.mean(l))

    plt.figure(figsize=(12,4))

    plt.subplot(121)
    plt.plot(mtl)
    plt.title('Training Accuracy' )

    plt.subplot(122)
    plt.plot(mvl)
    plt.title('Validation Accuracy')
    plt.show()