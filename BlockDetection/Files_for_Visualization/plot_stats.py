import pickle 
import numpy as np 
import matplotlib.pyplot as plt 



def plot_count_per_blocktype(counts):

    values = counts.values()
    names = counts.keys()

    print(counts['diamond'])
    plt.ylabel('# of collected blocks in dataset')
    plt.bar(names, values)
    plt.show()
    
def mean_steps_til_collection(las):
    means = []
    for key in las.keys():
        means.append(np.mean(las[key]))
    
    names = las.keys()
    plt.bar(names, means)
    plt.ylabel('Mean Steps between last attack and collection of block')
    plt.show()

def plot_hist(las):
    for key in las.keys():
        plt.hist(las[key], bins= [1, 10, 100, 1000, 10000])
        plt.title('Block type: ' + key)
        plt.xscale('log')
        plt.xlabel('Steps between last attack and collection of block')
        plt.show()

if __name__=='__main__':

    with open('CollectedBlockCounts.pickle', 'rb') as f:
        counts = pickle.load(f)
    
    with open('StepsFromAttack.pickle', 'rb') as f:
        las = pickle.load(f)
    
    plot_count_per_blocktype(counts)
    mean_steps_til_collection(las)
    plot_hist(las)