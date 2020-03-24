import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Linear, Softmax, Sigmoid, Conv1d, Sequential, ReLU
import torchvision
from torchvision import transforms
import data_process


def load_data(path):
    data = pd.read_csv(path)
    return data


def modify_csv(path):
    pre = pd.read_csv(path)
    pre['Survived'] = pd.Series(data=pre['Survived'], dtype=int)
    pre.to_csv(path_or_buf='prediction_dnn.csv', index=False)


def cal_ratio():
    data = load_data('titanic/train.csv')
    return data['Survived'].mean()


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.dense_layers = Sequential(
            Linear(7, 128),
            Sigmoid(),
            Linear(128, 64),
            Sigmoid(),
            Linear(64, 32),
            Sigmoid(),
            Linear(32, 16),
            Sigmoid(),
            Linear(16, 8),
            Sigmoid(),
            Linear(8, 4),
            Sigmoid()
        )
        self.output_layer = Sequential(
            Linear(4, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.output_layer(x)
        return x


class Titanic:

    age_scaler = StandardScaler()
    fare_scaler = StandardScaler()

    def __init__(self):
        pass

    def train(self):
        dtype = torch.float
        data_class = data_process.Preprocess()
        tmp = data_class.process_SibSp()
        data = load_data('data/train.csv')
        feature, label = self.data_pre_process(data, 'train')
        x = torch.tensor(feature, dtype=dtype)
        y = torch.tensor(label, dtype=dtype)

        # create model
        model = DNN()

        # Initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Initialize loss function
        criterion = nn.CrossEntropyLoss()

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        for i in range(100):
            # Forward pass: Compute predicted y by passing x to the model
            pre = model(x)

            # Compute and print loss
            loss = criterion(pre, y)

            print(i, loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), 'models/dnn.plk')

    def test(self):
        dtype = torch.float
        data = load_data('data/test.csv')
        feature, label = self.data_pre_process(data)
        x = torch.tensor(feature, dtype=dtype)

        model = DNN()
        model.load_state_dict(torch.load('models/dnn.plk'))
        pre = model(x)

        print(pre)


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
    # modify_csv('prediction_dnn.csv')
