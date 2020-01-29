import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Linear, Softmax, Sigmoid, Conv1d, Sequential, ReLU


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
            Softmax()
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

    def data_pre_process(self, data, mode='test'):
        # variables selection, normalization
        data['Sex'] = data['Sex'].replace(['male', 'female'], [1, 0])
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
        data['Embarked'] = data['Embarked'].fillna(3)
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        feature = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
        if mode == 'train':
            feature[:, 2] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 2], (-1, 1))), (-1))
            feature[:, 5] = np.reshape(self.fare_scaler.fit_transform(np.reshape(feature[:, 5], (-1, 1))), (-1))
            label = data['Survived']
            return feature, label
        else:
            feature[:, 2] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 2], (-1, 1))), (-1))
            feature[:, 5] = np.reshape(self.age_scaler.fit_transform(np.reshape(feature[:, 5], (-1, 1))), (-1))
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        data = load_data('data/train.csv')
        feature, label = self.data_pre_process(data, 'train')
        x = Variable(torch.tensor(feature))
        y = Variable(torch.tensor(label))

        # create model
        model = DNN()

        # Initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.0001)

        # Initialize loss function
        loss = nn.MSELoss()

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
            pre = model(x.float())

            # Compute and print loss
            lo = loss(pre, y)
            print(i, lo.item())

        torch.save(model.state_dict(), 'models/dnn.plk')


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    # modify_csv('prediction_dnn.csv')
