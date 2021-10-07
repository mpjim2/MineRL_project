import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import ANN 
import data_preparation
import numpy as np
import pickle
import os
import glob
import argparse
from datetime import datetime

class AttackPrediction:

    def __init__(self, batchsize, max_epochs, save_interval, save_dir, learning_rate, pooling, patchsizes):

        self.network = ANN.ActionPredictor()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.BCELoss() 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(self.device)
        
        self.train_loader, self.validation_loader = data_preparation.prepare_data(batchsize)
        self.max_epochs = max_epochs
        self.save_interval = save_interval
        
        self.save_dir = save_dir

        self.loss_stats = {'training' : [], 'validation' : []}
        self.cur_best = 100
        
    def train(self):

        for epoch in range(1, self.max_epochs+1):
            loss_train = []
            loss_validation = []

            for frame, label in self.train_loader:
                
                frame = frame.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                prediction = self.network(frame)
         
                loss = self.loss_fn(prediction, torch.unsqueeze(label, axis=1))
                loss_train.append(loss.item())

                loss.backward()
                self.optimizer.step()
                
            self.loss_stats['training'].append(loss_train)

            for frame, label in self.validation_loader:
                
                frame = frame.to(self.device)
                label = label.to(self.device)

                prediction = self.network(frame)
                
                loss = self.loss_fn(prediction, torch.unsqueeze(label, axis=1))
                loss_validation.append(loss.item())

            self.loss_stats['validation'].append(loss_validation)

            if (epoch % self.save_interval == 0):
                self.save_stats()
                if epoch >= 50:
                    self.save_model('MostRecent' + str(epoch))
                else:
                    self.save_model('MostRecent')

            if (loss < self.cur_best):
                self.save_model('BestPerforming')
                      
    def save_stats(self):

        PATH = self.save_dir + '/Statistics/'
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        
        save_stat(PATH, 'Validation_loss.pickle', self.loss_stats['validation'])
        save_stat(PATH, 'Training_loss.pickle', self.loss_stats['training'])

        self.loss_stats      = {'validation' : [], 
                                'training' : []}

    def save_model(self, name):
        PATH = self.save_dir + '/SavedModels/'
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        torch.save(self.network.state_dict(), PATH + name)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', required=True, help='int')
    parser.add_argument('--nepochs', required=True, help='int')
    parser.add_argument('--save_interval', required=False)
    parser.add_argument('--LR', required=True)
    parser.add_argument('--n_heads', required=True)
    parser.add_argument('--pooling', required=True)

    parser.add_argument('--save_path', required=False)
    opt = parser.parse_args()

    batchsize = int(opt.batchsize)
    max_epochs = int(opt.nepochs)
    save_interval = int(opt.save_interval)
    learning_rate = float(opt.LR)
    n_heads = int(opt.n_heads)
    pooling = int(opt.pooling)
    

    if opt.save_path is None:
        if pooling:
            save_dir = './batchsize=' + opt.batchsize  + '_maxepochs=' + opt.nepochs + '_learningrate=' + opt.LR + '_n_heads=' + opt.n_heads + '_pooling'
        else:
            save_dir = './batchsize=' + opt.batchsize  + '_maxepochs=' + opt.nepochs + '_learningrate=' + opt.LR + '_n_heads=' + opt.n_heads + '_nopooling'
    else:
        save_dir = opt.save_path
        
    if n_heads == 1:
        patchsizes = [16]
    if n_heads == 2:
        patchsizes = [16, 24]
    if n_heads == 3:
        patchsizes = [16, 24, 32]

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        now = datetime.now()
        day_str = now.strftime("%d-%m-%Y_%H:%M:%S")
        save_dir = save_dir + '_' + day_str
        os.mkdir(save_dir)

    algo = AttackPrediction(batchsize, max_epochs, save_interval, save_dir, learning_rate, pooling, patchsizes)

    algo.train()
