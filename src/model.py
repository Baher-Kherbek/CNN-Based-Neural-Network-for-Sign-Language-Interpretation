'''
Author:
    Baher Kher Bek
'''

import torch
from torch import nn

class SignLanguageInterpreter(torch.nn.Module):
    def __init__(self, InChannels, Classes):
        super(SignLanguageInterpreter, self).__init__()
        self.Relu = nn.ReLU()
        #1 : Convulution -> Relu Activation Function -> Batch Normalization -> Pooling
        self.Conv1 = nn.Conv2d(in_channels=InChannels, out_channels=20, kernel_size=(5, 5))
        self.BatchNorm1 = nn.BatchNorm2d(20)
        self.Pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #2 :  Convulution -> Relu Activation Function -> Dropout -> Batch Normalization -> Pooling
        self.Conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.Dropout = nn.Dropout(0.2)
        self.BatchNorm2 = nn.BatchNorm2d(50)
        self.Pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #3 : Fully Connected Layer -> Relu Activation Function
        self.FC1 = nn.Linear(in_features=800, out_features=200)

        #4 : SoftMax Classifier
        self.FC2 = nn.Linear(in_features=200, out_features= Classes)
        self.LogSoftMax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #Pass through the first layer
        x = self.Conv1(x)
        x = self.Relu(x)
        x = self.BatchNorm1(x)
        x = self.Pool1(x)

        # Pass through the second layer
        x = self.Conv2(x)
        x = self.Relu(x)
        x = self.Dropout(x)
        x = self.BatchNorm2(x)
        x = self.Pool2(x)

        #Flatten the input then pass through the FC layer
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = self.Relu(x)

        #Pass through the classifier to obtain output
        x = self.FC2(x)
        output = self.LogSoftMax(x)

        return output


if __name__ == '__main__':
    #testing the model
    model = SignLanguageInterpreter(InChannels=1, Classes=24)
    inp = torch.randn(16, 1, 28, 28)

    prediction = model(inp)
    print(prediction)


