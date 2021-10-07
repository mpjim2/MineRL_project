import minerl 
import numpy as np 
from tqdm import tqdm
import os
import pickle 

def iterate_episode(d, counts, last_attack_sums):
    current_states ,actions ,rewards ,next_states,_ = d
    
    num_frames = len(current_states['pov'])
    print(num_frames)
    frames = []

    inventory = current_states['inventory']
    
    print(rewards)
    last_attack = 0
    for i in range(num_frames-1):
        
        if actions['attack'][i+1] == 1:
            last_attack = 0
        else:
            last_attack +=1

        for key in counts.keys():
            if key != 'diamond':
                if inventory[key][i+1] > inventory[key][i]:
                    counts[key]+=1
                    last_attack_sums[key].append(last_attack)
    
    if actions['attack'][-1] == 1:
            last_attack = 0
        else:
            last_attack +=1

    if rewards[-1] == 1024:
        counts['diamond'] +=1
        last_attack_sums[key].append(last_attack)

    return counts, last_attack_sums


def extract_data(dataset='MineRLObtainDiamond-v0'):
    
    data = minerl.data.make(dataset, data_dir='../Data/')
    trajectories = data.get_trajectory_names()
    
    counts = {'log' : 0, 
              'cobblestone' : 0, 
              'coal' : 0, 
              'iron_ore' : 0,
              'dirt' : 0, 
              'diamond' : 0}

    last_attack_sums = {'log' : [], 
                        'cobblestone' : [], 
                        'coal' : [], 
                        'iron_ore' : [],
                        'dirt' : [],
                        'diamond' : []}

    for traj in tqdm(trajectories):
        d = data._load_data_pyfunc(os.path.join('../Data/MineRLObtainDiamond-v0', traj), -1, None)
        counts, last_attack_sums = iterate_episode(d, counts, last_attack_sums)
    
    return counts, last_attack_sums



if __name__=='__main__':
    
    c, las = extract_data()
    with open('./CollectedBlockCounts.pickle', 'wb') as f:
        pickle.dump(c, f)
    with open('./StepsFromAttack.pickle', 'wb') as f:
        pickle.dump(las, f)
    