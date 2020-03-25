import pandas as pd
import torch
from torch import nn, optim
from torch.nn import Linear, Sigmoid, Sequential, ReLU, Dropout
from data_process import PreProcess


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.dense_layers = Sequential(
            Linear(24, 256),
            ReLU(),
            Dropout(),
            Linear(256, 32),
            ReLU(),
            Dropout()
        )
        self.output_layer = Sequential(
            Linear(32, 1)
        )

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.output_layer(x)
        return x


class Titanic:

    def __init__(self):
        pass

    def train(self):
        dtype = torch.float
        process = PreProcess()
        process.load_data('data/train.csv')
        feature, label = process.merge_data()
        x = torch.tensor(feature, dtype=dtype)
        y = torch.tensor(label, dtype=dtype)

        # create model
        model = DNN()

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Initialize loss function
        criterion = nn.MSELoss()

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        for i in range(1000):
            # Forward pass: Compute predicted y by passing x to the model
            pre = model(x)

            # Compute and print loss
            loss = criterion(pre, y)

            print(i, loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pre = model(x)
        result = []

        for item in pre:
            if item > 0.5:
                result.append(1)
            else:
                result.append(0)

        cnt = 0
        for prediction, ground_truth in zip(result, y):
            if prediction == ground_truth:
                cnt += 1
        acc = cnt / len(result)
        print(acc)

        torch.save(model.state_dict(), 'models/dnn.plk')

    def test(self):
        dtype = torch.float
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, pid = process.merge_data('test', if_one_hot=True)
        x = torch.tensor(feature, dtype=dtype)

        model = DNN()
        model.load_state_dict(torch.load('models/dnn.plk'))
        pre = model(x)

        result = []

        for item in pre:
            if item > 0.5:
                result.append(1)
            else:
                result.append(0)

        prediction = pd.Series((i for i in result), name='Survived')

        submission = pd.concat([pid, prediction], axis=1)

        submission.to_csv(path_or_buf='data/gender_submission.csv', index=False)


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
    # modify_csv('prediction_dnn.csv')
