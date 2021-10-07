import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='VGG | DCGAN')
    opt =parser.parse_args()
    model = opt.model

    PATH_t = './' + model + '_batchsize64_datasetSize400000_MineRLTreechop-v0_MSE/Statistics/Training_loss.pickle'
    PATH_v = './' + model + '_batchsize64_datasetSize400000_MineRLTreechop-v0_MSE/Statistics/Validation_loss.pickle'
    with open(PATH_t, 'rb') as file:
        tl = pickle.load(file)

    with open(PATH_v, 'rb') as file:
        vl = pickle.load(file)

    print(len(tl))
    print(len(vl))
    mtl = []
    mvl = []

    for s in tl:
        for l in s:
            print("\t" ,len(l))
            mtl.append(np.mean(l))

    for s in vl:
        for l in s:
            print("\t" ,len(l))
            mvl.append(np.mean(l))

    plt.figure(figsize=(12,4))

    plt.subplot(121)
    plt.plot(mtl)
    plt.title('Training Loss ' + model)

    plt.subplot(122)
    plt.plot(mvl)
    plt.title('Validation Loss ' + model)
    plt.show()