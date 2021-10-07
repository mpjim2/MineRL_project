import pickle 
import numpy as np 




if __name__ == '__main__':
    
    with open('./frames_labels.pickle', 'rb') as f:
        frames, labels = pickle.load(f)

    with open('./CollectedBlockCounts.pickle', 'rb') as f:
        c = pickle.load(f)

    assert len(frames) == len(labels)
    
    for key in c.keys():
        ids = [i for i, x in enumerate(labels) if x == key]

        filename = key + '_frames.pickle'
        key_frames = [frames[i] for i in ids]

        with open('./' + filename, 'wb') as f:
            pickle.dump(key_frames, f)
