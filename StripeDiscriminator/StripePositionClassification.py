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
import argparse
from datetime import datetime

class StripePositionClassification:

    def __init__(self, batchsize= None, max_epochs=None, save_interval=None, save_dir=None):

        self.network = model.StripeClassifier()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #if all hyperparameters for training are none: load a saved model for testing
        if batchsize = None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0001)
            self.loss_fn = torch.nn.CrossEntropyLoss() 
            
           
            self.network.to(self.device)
            
            
            self.train_loader, self.validation_loader = Data_preparation.prepare_data(batchsize)
            self.max_epochs = max_epochs
            self.save_interval = save_interval
            
            self.save_dir = save_dir

            self.bestacc = 0

            self.loss_stats = {'training' : [], 'validation' : []}
            self.acc_stats = {'training' : [], 'validation' : []}
        else: 
            
            self.network.load_state_dict(torch.load(save_dir + '/SavedModels/MostRecent100', map_location=self.device))
            self.network.to(self.device)


    def test(self, frame): 

        prediction = self.network.predict(frame).detach().numpy()

        return prediction

    def train(self):

        for epoch in range(1, self.max_epochs+1):
            loss_train = []
            loss_validation = []

            acc_train =[]
            acc_validation = []

            for stripe, label in self.train_loader:
                stripe = stripe.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                pred_probs = self.network(stripe)
                
                predicted_labels = torch.argmax(pred_probs, dim=1)
                acc = accuracy(predicted_labels, label)
                acc_train.append(acc)

                loss = self.loss_fn(pred_probs, label)
                loss_train.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                
            self.loss_stats['training'].append(loss_train)
            self.acc_stats['training'].append(acc_train)

            for stripe, label in self.validation_loader:
                
                stripe = stripe.to(self.device)
                label = label.to(self.device)

                pred_probs = self.network(stripe)
                
                predicted_labels = torch.argmax(pred_probs, dim=1)
                acc = accuracy(predicted_labels, label)
                acc_validation.append(acc)

                loss = self.loss_fn(pred_probs, label)
                loss_validation.append(loss.item())

            if np.mean(acc_validation) > self.bestacc:
                self.save_model('BestAccuracy')
                self.bestacc = np.mean(acc_validation)
                
            self.loss_stats['validation'].append(loss_validation)
            self.acc_stats['validation'].append(acc_validation)

            if (epoch % self.save_interval == 0):
                self.save_stats()
                if epoch >= 50:
                    self.save_model('MostRecent' + str(epoch))
                else:
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

        save_stat(PATH, 'Validation_accuracy.pickle', self.acc_stats['validation'])
        save_stat(PATH, 'Training_accuracy.pickle', self.acc_stats['training'])

        self.acc_stats       = {'validation' : [], 
                                'training' : []}
        

    def save_model(self, name):
        PATH = self.save_dir + '/SavedModels/'
        try:
            os.mkdir(PATH)
        except FileExistsError:
            pass
        torch.save(self.network.state_dict(), PATH + name)


def accuracy(prediction, labels):

    prediction = prediction.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    correct = len(np.argwhere(prediction == labels))

    return correct / len(prediction)

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


#Main method starts training with given Hyperparameters
if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', required=True, help='int')
    parser.add_argument('--nepochs', required=True, help='int')
    #parser.add_argument('--datasetSize', required=True, help='int')
    parser.add_argument('--save_interval', required=False)

    opt = parser.parse_args()

    batchsize = int(opt.batchsize)
    max_epochs = int(opt.nepochs)
    #datasetSize = int(opt.datasetSize)
    save_interval = int(opt.save_interval)

    save_dir = './batchsize=' + opt.batchsize  + '_maxepochs=' + opt.nepochs

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        now = datetime.now()
        day_str = now.strftime("%d-%m-%Y_%H:%M:%S")
        save_dir = save_dir + '_' + day_str
        os.mkdir(save_dir)

    print('INITIALIZE PARAMETERS')
    algo = StripePositionClassification(batchsize, max_epochs, save_interval, save_dir)
    print('INIT OK!')

    print('START TRAINING')
    algo.train()