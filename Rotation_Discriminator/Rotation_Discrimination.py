import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import models 
import Data_preparation
import numpy as np
import pickle
import os
import glob
import argparse



class Rotation_Discrimination:

    def __init__(self, model, SAVEDIR, mode='Test'; dataset=None, lossfn=None, MAX_EPOCHS=None, BATCH_SIZE=None, DATASIZE=None, SAVE_INTERVAL=None):

        if model == 'DCGAN':
            self.network = models.DCGAN_Discriminator()
        if model == 'VGG':
            self.network = models.VGG_Discriminator()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == 'Train': 
            if lossfn == 'BCE':
                self.loss = torch.nn.BCELoss()
            if lossfn == 'MSE':
                self.loss = torch.nn.MSELoss()

            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
            
            self.network.to(self.device)

            self.SAVE_INTERVAL = SAVE_INTERVAL
            self.SAVEDIR = SAVEDIR
            self.BATCH_SIZE = BATCH_SIZE
            self.train_loader, self.validation_loader = Data_preparation.prepare_data(dataset=dataset, samplecount = DATASIZE, batch_size=self.BATCH_SIZE)

            self.MAX_EPOCHS = MAX_EPOCHS

            self.best_performance = -999

            self.loss_statistics = {'validation' : [], 
                                    'training' : []}
            
            self.acc_statistics = {'validation' : [],
                                    'training' : []}
        else: 
            self.network.load_state_dict(torch.load(SAVEDIR, map_location=device))
            self.network.to(self.device)
    

    def test(self, frame : np.ndarray) -> np.ndarray: 
        '''returns the probability that a given frame is rotated'''
        probability = self.network(frame).detach().numpy()

        return probability

    def train(self): 
        
        for epoch in range(1, self.MAX_EPOCHS+1):
            loss_train = []
            loss_val = []
            acc_train = []
            acc_val = []
            
            for frame, label in self.train_loader:
                if self.gpu:
                    frame = frame.to(self.device)
                    label =label.to(self.device)
                self.optimizer.zero_grad()

                pred_probs = self.network(frame)

                if self.recordAcc:
                    pred_acts  = (pred_probs >= 0.5).detach().numpy().astype(int)
                    correct = np.where((pred_acts - label.numpy().astype(int)) == 0)
                    acc = len(correct[0]) / self.BATCH_SIZE
                    acc_train.append(acc)

                loss = self.loss(pred_probs.float(), label.float())
                loss.backward()

                self.optimizer.step()
                loss_train.append(loss.item())
             
            self.loss_statistics['training'].append(loss_train)
            self.acc_statistics['training'].append(acc_train)   
            
            for frame, label in self.validation_loader:
                
                if self.gpu:
                    frame = frame.to(self.device)
                    label =label.to(self.device)
                pred_probs = self.network(frame)
                
                if self.recordAcc:
                    pred_acts  = (pred_probs >= 0.5).detach().numpy().astype(int) 
                    correct = np.where((pred_acts - label.numpy().astype(int)) == 0)
                    acc = len(correct[0]) / self.BATCH_SIZE
                    acc_val.append(acc)

                loss = self.loss(pred_probs.float(), label.float())
                loss_val.append(loss.item())
            
            
            self.acc_statistics['validation'].append(acc_val)
            performance = np.mean(acc_val)
           
            if (performance >= self.best_performance):
                print(performance)
                self.best_validation = np.mean(acc_val)
                self.save_model(self.SAVEDIR + '/Saved_Models/', 'BestPerforming')

            self.loss_statistics['validation'].append(loss_val)
            
            if epoch % self.SAVE_INTERVAL == 0:
                self.save_model(self.SAVEDIR + '/Saved_Models/', 'MostRecent')
                self.save_stats(self.SAVEDIR + '/Statistics/', epoch)

    def save_model(self, PATH, name):
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        torch.save(self.network.state_dict(), PATH + name)

    def save_stats(self, PATH, epoch):
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        
        save_stat(PATH, 'Validation_loss.pickle', self.loss_statistics['validation'])
        save_stat(PATH, 'Training_loss.pickle', self.loss_statistics['training'])

        self.loss_statistics = {'validation' : [], 
                                'training' : []}
        

        save_stat(PATH, 'Validation_accuracy.pickle', self.acc_statistics['validation'])
        save_stat(PATH, 'Training_accuracy.pickle', self.acc_statistics['training'])
        
        self.acc_statistics = {'validation' : [],
                            'training' : []}
        
    
def save_stat(PATH, name, stat):
    
    try:
        with open(PATH + name, 'rb') as f:
            stat_sofar = pickle.load(f)
        stat_sofar.append(stat)
    except FileNotFoundError:
        stat_sofar = [stat]
    except EOFError:
        stat_sofar = [stat]

    with open(PATH + name, 'wb') as f:
        pickle.dump(stat_sofar, f)
    
if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='Train | Test')
    parser.add_argument('--batchsize', required=False, help='int')
    parser.add_argument('--nepochs', required=False, help='int')
    parser.add_argument('--datasetSize', required=False, help='int <400k')
    parser.add_argument('--lossfun', required=False, help='BCE | MSE')
    parser.add_argument('--save_interval', required=False)
    parser.add_argument('--model', required=False, help='VGG | DCGAN')
    parser.add_argument('--savedir', required=False, help='Specify the directory where trained models should be saved')
    parser.add_argument('--gpu', required=False, help='0 | 1')
    parser.add_argument('--dataset', required=False, help='MineRLTreechop-v0 | MineRLNavigate-v0')
    opt = parser.parse_args()
    
    mode = opt.mode

    if mode == 'Train': 
        batchsize = int(opt.batchsize)
        nepochs = int(opt.nepochs)
        dataSize = int(opt.datasetSize)
        lossfn = opt.lossfun
        dataset = opt.dataset

        if opt.save_interval is None:
            save_interval = 5
        else:
            save_interval = int(opt.save_interval)
        if opt.model is None:
            model = 'DCGAN'
        else:
            model = opt.model

        if opt.savedir is None:
            savedir = './' + model + '_batchsize' + opt.batchsize + '_datasetSize' + opt.datasetSize + '_' + opt.dataset + '_' + opt.lossfun
        else:
            savedir = opt.savedir

        try:
            os.mkdir(savedir)
        except FileExistsError:
            pass

        print(savedir)
        ac = Rotation_Discrimination(model, savedir, mode, dataset, lossfn, nepochs, batchsize, dataSize, save_interval)
        ac.train()
