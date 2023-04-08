'''
Author:
    Baher Kher Bek
'''

import torch
from torch import nn
from dataset import loadDataset
from model import SignLanguageInterpreter
import pickle

if __name__ == '__main__':
    # Load Dataset
    TrainingData, TestingData = loadDataset(path='/Users/baherkherbek/Desktop/pyt/model_1/Dataset', BatchSize=64)

    # Load Model
    model = SignLanguageInterpreter(InChannels=1, Classes=25)
    if torch.cuda.is_available():
        model.cuda()

    # Set training Paramaters
    Epochs = 40
    learning_rate = 0.00001
    Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossFunction = nn.CrossEntropyLoss()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    History = {
        'Epoch': [],
        'TrainingLoss': [],
        'Accuracy': []
    }

    flag = None
    # Train and Evaluate
    for epoch in range(Epochs):
        flag = True
        for (Batch, Target) in TrainingData:
            model.train()
            (Batch, Target) = (Batch.to(device), Target.to(device))
            Batch = Batch.unsqueeze(dim=1)  # [64, 28, 28] ---> [64, 1, 28, 28]

            # Get prediction for the batch and compute training loss
            prediction = model(Batch)
            TrainingLoss = lossFunction(prediction, Target)
            Optimizer.zero_grad()

            # compute gradient
            TrainingLoss.backward()

            # update weights
            Optimizer.step()

            # Evaluate Accuracy testing data
            count = 0
            NumTrueClassifications = 0
            Accuracy = None

            if flag:
                flag = False

                # Compute Testing Loss Accuracy of model through each epoch on Unseen data
                for (Batch, Target) in TestingData:
                    model.eval()
                    (Batch, Target) = (Batch.to(device)), (Target.to(device))
                    Batch = Batch.unsqueeze(dim=1)
                    prediction = model(Batch)
                    count += len(Batch)
                    Classification = torch.argmax(prediction, dim=1)
                    NumTrueClassifications += sum(Classification == Target).item()

                Accuracy = NumTrueClassifications * 100 / count

                History['Epoch'].append(epoch)
                History['TrainingLoss'].append((TrainingLoss.cpu()).detach().numpy())
                History['Accuracy'].append(Accuracy)
                print(f'Epoch: {epoch}/{Epochs} | Loss: {TrainingLoss} | Accuracy: {Accuracy}%')

    # Save Model and Data History
    torch.save(model, './SignLanguageModel')
    fileHandle = open('./Training Data', 'ab')
    pickle.dump(History, fileHandle)
    fileHandle.close()
