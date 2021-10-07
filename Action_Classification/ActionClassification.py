import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import model 
import Data_preparation
import numpy as np
import pickle
import os
import glob



class Action_Classification:

    def __init__(self, MAX_EPOCHS: int, BATCH_SIZE : int, model_path=None, load_saved_weights=False):
        
        self.network = model.action_classifier()
        self.loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        self.BATCH_SIZE = BATCH_SIZE
        self.train_loader, self.validation_loader, self.test_loader = Data_preparation.prepare_data(samplecount = 200000, batch_size=self.BATCH_SIZE)

        self.MAX_EPOCHS = MAX_EPOCHS

        #Dictionaries to save loss and accuracy during training and validation
        self.loss_statistics = {'validation' : [], 
                                'training' : []}
        self.acc_statistics = {'validation' : [],
                                'training' : []}


        if load_saved_weights and model_path is not None:
            self.network.load_state_dict(torch.load(model_path, map_location=device))

    #Training Loop
    def train(self): 

        for i in range(self.MAX_EPOCHS):
            loss_train = []
            loss_val = []
            acc_train = []
            acc_val = []
            
            for f1, f2, action in self.train_loader:
                
                self.optimizer.zero_grad()

                pred_probs = self.network(f1, f2)
                pred_acts  = (pred_probs >= 0.5).detach().numpy().astype(int)
                
                #Compute accuracy
                correct = np.where((pred_acts - action.numpy().astype(int)) == 0)
                acc = len(correct[0]) / self.BATCH_SIZE
                acc_train.append(acc)

                loss = self.loss(pred_probs, action)
                loss.backward()

                self.optimizer.step()
                loss_train.append(loss.item())
             
            self.loss_statistics['training'].append(loss_train)
            self.acc_statistics['training'].append(acc_train)   
            
            for f1, f2, action in self.validation_loader:
                
                pred_probs = self.network(f1, f2)
                pred_acts  = (pred_probs >= 0.5).detach().numpy().astype(int)
                
                #compute accuracy
                correct = np.where((pred_acts - action.numpy().astype(int)) == 0)
                acc = len(correct[0]) / self.BATCH_SIZE

                loss = self.loss(pred_probs, action)
                loss_val.append(loss.item())
                acc_val.append(acc)

            self.loss_statistics['validation'].append(loss_val)
            self.acc_statistics['validation'].append(acc_val)

    def test(self, frame1 : np.ndarray, frame2 : np.ndarray) -> np.ndarray:
        '''Test function; returns the probability for the (trained) action given two successive states'''
        action_probability = self.network(frame1, frame2)
        action_probability = action_probability.detach().numpy()

        return action_probability

    def save(self):
        try:
            os.mkdir('./Statistics')
        except FileExistsError:
            pass
        
        number = len(glob.glob('./Statistics/*.pickle'))

        with open('./Statistics/losses_{}.pickle'.format(int(number/2)), 'wb') as f:
            pickle.dump(self.loss_statistics, f)
        with open('./Statistics/accuracies_{}.pickle'.format(int(number/2)), 'wb') as f:
            pickle.dump(self.acc_statistics, f)

        try:
            os.mkdir('./Saved_models')
        except FileExistsError:
            pass
        
        number = len(glob.glob('./Saved_models/*'))
        
        torch.save(self.network.state_dict(), './Saved_models/model_{}'.format(number))    


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    ac = Action_Classification(100, 128)
    ac.train()
    ac.save()