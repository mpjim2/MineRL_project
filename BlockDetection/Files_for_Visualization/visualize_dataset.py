import pickle 
import matplotlib.pyplot as plt 
import random
import numpy as np 
from PIL import Image

if __name__ == '__main__':

    path = './'
    keys = ['coal', 'cobblestone', 'log', 'iron_ore', 'diamond', 'dirt', 'misc']

    #keys = ['coal']
    for key in keys: 
        print(key)
        filename = './' + key + '_frames.pickle'

        with open(filename, 'rb') as f:
            frames = pickle.load(f)
        
        print(len(frames))
        if len(frames) > 1024:
            rnd_ids = np.random.choice(len(frames), 1024)
            frames = [frames[i] for i in rnd_ids]
        else:
            random.shuffle(frames)

        visu = np.zeros((32*64, 32*64, 3), dtype=np.uint8)

        for n, frame in enumerate(frames):
            row = int(n/32)
            col = n%32

            visu[row*64 : (row+1)*64, col*64 : (col+1)*64, :] = frame

        im = Image.fromarray(visu)
        im.save(path + key + '_frames.png')
        
  

