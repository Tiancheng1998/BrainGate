import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda')


output_size = 1
hidden_size1 = 100
hidden_size2 = 50

# The following classes are pytorch models for various architectures.


class TwoLayerDNN(torch.nn.Module):
    def __init__(self, win, channel, rates_and_power):
        super(TwoLayerDNN, self).__init__()
        if rates_and_power:
            input_size = channel*2*win
        else:
            input_size = channel*win
        self.layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.layer2 = torch.nn.Linear(hidden_size1, output_size)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        return self.sig(x)


class ThreeLayerDNN(torch.nn.Module):
    def __init__(self, win, channel, rates_and_power):
        super(ThreeLayerDNN, self).__init__()
        if rates_and_power:
            input_size = channel*2*win
        else:
            input_size = channel*win
        input_size = channel*win
        self.layer1 = torch.nn.Linear(input_size, hidden_size1)
        self.layer3 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.layer2 = torch.nn.Linear(hidden_size2, output_size)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        return self.sig(x)



class ConvDNN(nn.Module):
    def __init__(self, win, channel):
        super().__init__()
        self.ConvLayer = nn.Conv1d(channel*2, channel*2*5, 15)
        self.pool = nn.AvgPool1d(3)
        self.dense = None
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.ConvLayer(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        if self.dense is None:
            self.dense = nn.Linear(x.shape[1], 1)
            self.dense = self.dense.cuda()
        x = self.dense(x)
        return self.sig(x)




class LSTMmodel(nn.Module):
    def __init__(self, win, channel):
        super().__init__()
        self.hidden = win * channel // 4
        self.lstm_unit = nn.LSTM(channel*2, self.hidden, batch_first=True)
        self.V = nn.Parameter(torch.empty(self.hidden, 1).normal_(std=0.1))
        self.b = nn.Parameter(torch.empty(1).normal_(std=0.1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm_unit(x)
        h = torch.squeeze(h)
        return self.sig(h@self.V + self.b)

