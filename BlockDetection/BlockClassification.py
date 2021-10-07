import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import classifier
import data_preparation
import numpy as np
import pickle
import os
import glob
import argparse
from datetime import datetime

class BlockClassification:
    '''Implements all necessary functions for training and testing the Classification of blocktypes the agent is looking at;
    To test an already trained model initialize this class with only the save_dir as parameter'''
    def __init__(self, batchsize=None, max_epochs=None, save_interval=None, save_dir=None, learning_rate=None, samplesPerclass=None):
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if batchsize is not None:
            
            self.batchsize = batchsize
            self.train_loader, self.validation_loader, self.block2id = data_preparation.prepare_data(batchsize, samplesPerclass)

            #save  the ID for each blocktype as a dictionary
            with open('./BlockIDdict.pickle', 'wb') as f:
                pickle.dump(self.block2id, f)

            self.network = classifier.BlockClassifier(len(self.block2id))

            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
            self.loss_fn = torch.nn.CrossEntropyLoss() 
            
            
            self.network = self.network.to(self.device)
            
            self.max_epochs = max_epochs
            self.save_interval = save_interval
            
            self.save_dir = save_dir

            self.loss_stats = {'training' : [], 'validation' : []}
            self.acc_stats = {'training': [], 'validation' : []}
            self.cur_best = 0
        
        else:
            
            with open('./BlockIDdict.pickle', 'wb') as f:
                self.block2id = pickle.load(f)

            self.network = classifier.BlockClassifier(len(self.block2id))
            if save_dir is not None:
                self.network.load_state_dict(torch.load(save_dir + '/Saved_models/HighestAccuracy', map_location=self.device))
            else:
                self.network = self.network.to(self.device)

    
    def test(self, frame): 
        
        block_probabilities = self.network(frame).detach().numpy()

        return block_probabilities
    
    
    def train(self):
        '''Training loop''' 
        for epoch in range(1, self.max_epochs+1):
            loss_train = []
            loss_validation = []

            acc_train = []
            acc_validation = []
            for frame, label in self.train_loader:
                
                frame = frame.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                logits = self.network(frame)

                accuracy = self.compute_accuracy(logits, label)


                loss = self.loss_fn(logits, label)
                loss_train.append(loss.item())
                acc_train.append(accuracy)
                loss.backward()
                self.optimizer.step()
                
            self.loss_stats['training'].append(loss_train)
            self.acc_stats['training'].append(acc_train)

            for frame, label in self.validation_loader:
                
                frame = frame.to(self.device)
                label = label.to(self.device)

                logits = self.network(frame)
                
                accuracy = self.compute_accuracy(logits, label)

                
                loss = self.loss_fn(logits, label)
                loss_validation.append(loss.item())
                acc_validation.append(accuracy)
            
            self.loss_stats['validation'].append(loss_validation)
            self.acc_stats['validation'].append(acc_validation)

            #save model if validation accuracy is higher than previous max
            if np.mean(acc_validation) >= self.cur_best:
                self.save_model('HighestAccuracy')
                print('Highest Accuracy at Epoch {}: {}'.format(epoch, np.mean(acc_validation)))
                self.cur_best = np.mean(acc_validation)
            
            #save to another file every save_interval episodes
            if (epoch % self.save_interval == 0):
                self.save_stats()
                self.save_model('MostRecent')    
                      
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
        '''saves the Model weights'''
        PATH = self.save_dir + '/SavedModels/'
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        torch.save(self.network.state_dict(), PATH + name)

    def compute_accuracy(self, logits, label):
        '''Computes the accuracy in a batch'''
        prediction = torch.argmax(logits, dim=1)

        correct = ((prediction - label) == 0).nonzero().shape[0]
        
        accuracy = correct / self.batchsize
        return accuracy


def save_stat(PATH, name, stat):
    '''Save statistics to a specific path; if the file already exists, load the file and append the data to it and save again'''    
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

    '''Main function starts the training loop with the given hyperparameters'''
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', required=True, help='int')
    parser.add_argument('--nepochs', required=True, help='int')
    parser.add_argument('--save_interval', required=False)
    parser.add_argument('--LR', required=True)
    parser.add_argument('--samplesPerclass', required=True)

    parser.add_argument('--save_path', required=False)
    opt = parser.parse_args()

    batchsize = int(opt.batchsize)
    max_epochs = int(opt.nepochs)
    save_interval = int(opt.save_interval)
    learning_rate = float(opt.LR)
    spc = int(opt.samplesPerclass)

    if opt.save_path is None:
        save_dir = './batchsize=' + opt.batchsize  + '_maxepochs=' + opt.nepochs + '_learningrate=' + opt.LR + '_SamplesPerClass=' + opt.samplesPerclass
    else:
        save_dir = opt.save_path
        

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        now = datetime.now()
        day_str = now.strftime("%d-%m-%Y_%H:%M:%S")
        save_dir = save_dir + '_' + day_str
        os.mkdir(save_dir)

    algo = BlockClassification(batchsize, max_epochs, save_interval, save_dir, learning_rate, spc)

    algo.train()