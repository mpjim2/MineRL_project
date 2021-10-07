import minerl 
import numpy as np 
from tqdm import tqdm
import os
import pickle 

def iterate_episode(d, counts, last_attack_sums, frames, labels, collect_misc):
    '''iterates a given episode; Estimates the block the agent is looking at based on the blocks collected'''
    current_states ,actions ,rewards ,next_states,_ = d
    
    num_frames = len(current_states['pov'])
    
    inventory = current_states['inventory']
    povs      = current_states['pov']

    #Size of the window of frames that are labeled as a specific class, after that blockw as collected; e.g. If a dirt block was collected at state t
    #all frames from t-10 to t-3 are labelled as 'dirt'
    frame_window = {'dirt' : [10, 3], 'log' : [10, 3], 'cobblestone' : [15, 5], 'coal' : [25, 5], 'iron_ore' : [30, 5], 'diamond' : [50, 7], 'misc' : [15, 3]}
    last_attack = 0

    #counter on how many misc frames are already collected
    misc_counter = 0
    
    for i in range(num_frames-1):
        
        #memorize when the agent last attacked(and therefore focused a specific block)
        if actions['attack'][i+1] == 1:
            last_attack = 0
        else:
            last_attack +=1

            if collect_misc:
                #'misc' frames are random frames from the Navigate datasets; dont collect more than 50,000
                if np.random.uniform(size=1) >= 0.5 and misc_counter < 50000:
                    frames.append(povs[i])
                    labels.append('misc')
                    misc_counter += 1

        #iterate over all possible items in the inventory
        for key in inventory.keys():
            #dont consider these items in the inventory, as they can only be aquired by crafting 
            if not key in ['wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 
                           'iron_pickaxe', 'planks', 'stick', 'furnace', 'iron_ingot', 'torch', 'stone']:
                

                #if the count of that item has increased since the last state, label respective frames (based on the frame window for that class)
                if inventory[key][i+1] > inventory[key][i]:
                    if key in counts.keys():
                        ck = key
                    
                        counts[ck]+=1
                        last_attack_sums[ck].append(last_attack)
                        
                        #based on the last_attack -counter and the frame window, label frames
                        #if last attack was too long ago, dont consider these blocks
                        if last_attack <= 100:
                            for j in range(i-(last_attack+frame_window[ck][0]), i-(last_attack+frame_window[ck][1])):
                                frames.append(povs[j])
                                labels.append(ck)
        
    if actions['attack'][-1] == 1:
        last_attack = 0
    else:
        last_attack +=1

    #no slot for diamond in the inventory; therefore use reward of 1024 as indicator for collecting one
    if rewards[-1] == 1024:
        counts['diamond'] +=1
        last_attack_sums['diamond'].append(last_attack)
        for j in range(len(povs)-(last_attack+frame_window['diamond'][0]), len(povs)-(last_attack+frame_window['diamond'][1])):
            frames.append(povs[j])
            labels.append('diamond')

    return counts, last_attack_sums, frames, labels


def extract_data(datasets):
    '''iterates all episodes of the given Datasets and collects the labels and frames'''
    counts = {'log' : 0, 
                'cobblestone' : 0, 
                'coal' : 0, 
                'iron_ore' : 0,
                'dirt' : 0, 
                'diamond' : 0,
                'misc' : 0}

    last_attack_sums = {'log' : [], 
                        'cobblestone' : [], 
                        'coal' : [], 
                        'iron_ore' : [],
                        'dirt' : [],
                        'diamond' : [], 
                        'misc' : []}
    
    frames = []
    labels = []

    faulty_eps = 0
    for dataset in datasets:
        
        if dataset  in ['MineRLNavigateDense-v0', 'MineRLNavigateExtremeDense-v0', 'MineRLNavigateExtreme-v0']:
            collect_misc = True
        else:
            collect_misc = False
        data = minerl.data.make(dataset, data_dir='../MINERLDATA/')
        trajectories = data.get_trajectory_names()

        for traj in tqdm(trajectories):
            path = '../MINERLDATA/' + dataset
    
            d = data._load_data_pyfunc(os.path.join(path, traj), -1, None)
            try:
                counts, last_attack_sums, frames, labels = iterate_episode(d, counts, last_attack_sums, frames, labels, collect_misc)
            except Exception as e:
                print(e)
                faulty_eps +=1
    
    print(faulty_eps)
    return counts, last_attack_sums, frames, labels



if __name__=='__main__':
    '''Collect all data and save them as pickle files to the disk'''

    #'MineRLTreechop-v0',  has no inventory therefore not included
    
    dataset_names = ['MineRLObtainDiamond-v0', 'MineRLNavigateDense-v0', 'MineRLNavigateExtremeDense-v0', 'MineRLNavigateExtreme-v0', 
                     'MineRLNavigate-v0', 'MineRLObtainDiamondDense-v0','MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0']
    
    c, las, frames, labels = extract_data(dataset_names)
    
    #statistics about blocktypes and steps taken from last attacking and collecting a block of a specific type
    with open('./CollectedBlockCounts.pickle', 'wb') as f:
        pickle.dump(c, f)
    with open('./StepsFromAttack.pickle', 'wb') as f:
        pickle.dump(las, f)

    data = (frames, labels)
    with open('./frames_labels.pickle', 'wb') as f:
        pickle.dump(data, f)


    