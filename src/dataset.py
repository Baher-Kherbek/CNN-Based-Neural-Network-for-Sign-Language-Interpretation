'''
Author:
    Baher Kher Bek
'''

import torch
import pandas
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def loadDataset(path='./Dataset', BatchSize=64):
    dataset = pandas.read_csv(path+'/train.csv')

    xtrain = (dataset.drop('label', axis=1)).values / 255
    xtrain = xtrain.reshape(len(xtrain), 28, 28)
    ytrain = dataset['label'].values

    inputs = torch.tensor(xtrain, dtype=torch.float32)
    targets = torch.tensor(ytrain, dtype=torch.int64)
    Training_data = TensorDataset(inputs, targets)
    Training_data = DataLoader(Training_data, batch_size=BatchSize, shuffle=True)

    dataset = pandas.read_csv(path+'/test.csv')
    xtest = (dataset.drop('label', axis=1)).values / 255
    xtest = xtest.reshape(len(xtest), 28, 28)
    ytest = dataset['label'].values

    inputs = torch.tensor(xtest, dtype=torch.float32)
    targets = torch.tensor(ytest, dtype=torch.int64)
    Testing_data = TensorDataset(inputs, targets)
    Testing_data = DataLoader(Testing_data, batch_size=BatchSize, shuffle=True)

    return Training_data, Testing_data

if __name__ == '__main__':
    TrainingData, TestingData = loadDataset(path='/Users/baherkherbek/Desktop/pyt/model_1/Dataset')
    print(f'Training Dataloader: {TrainingData} \n Testing Dataloader : {TestingData}')