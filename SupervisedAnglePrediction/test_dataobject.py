import minerl 
import numpy as np 
import os
import pickle

data = minerl.data.make('MineRLTreechop-v0', data_dir='../Data/')
trajectories = data.get_trajectory_names()

def iterate_episode(d):
    current_states ,actions ,_ ,next_states,_ = d

    assert len(current_states['pov']) == len(actions['camera']) 

    num_frames = len(current_states['pov'])

    frames = []
    angles = []

    cur_angle = 0
    for i in range(num_frames):
        angles.append(cur_angle)
        frames.append(current_states['pov'][i])
        cur_angle -= actions['camera'][i][0]
        if cur_angle > 90:
            cur_angle = 90
        if cur_angle < -90:
            cur_angle = -90

    frames = np.asarray(frames)
    angles = np.asarray(angles)

    return frames, angles


for traj in trajectories:
    d = data._load_data_pyfunc(os.path.join('../Data/MineRLTreechop-v0', traj), -1, None)
    f, a = iterate_episode(d)
    
    break