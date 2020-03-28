import pandas as pd
import torch
from torch import nn, optim
from torch.nn import Linear, Sequential, ReLU, Dropout
from data_process import PreProcess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.dense_layers = Sequential(
            Linear(7, 256),
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
        feature, label = process.merge_data(mode='train', if_one_hot=False, continuous=True)

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

        for i in range(10000):
            # Forward pass: Compute predicted y by passing x to the model
            feature_train, feature_validation, label_train, label_validation = train_test_split(feature, label, train_size=0.8, shuffle=True)
            x_train = torch.tensor(feature_train, dtype=dtype)
            y_train = torch.tensor(label_train, dtype=dtype)
            x_validation = torch.tensor(feature_validation, dtype=dtype)
            y_validation = torch.tensor(label_validation, dtype=dtype)

            pre_train = model(x_train)
            pre_validation = model(x_validation)

            # Compute and print loss
            loss = criterion(pre_train, y_train)
            val_loss = criterion(pre_validation, y_validation)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result = []

            for item in pre_train:
                if item > 0.5:
                    result.append(1)
                else:
                    result.append(0)

            cnt = 0
            for prediction, ground_truth in zip(result, y_train):
                if prediction == ground_truth:
                    cnt += 1
            acc = cnt / len(result)

            result = []

            for item in pre_validation:
                if item > 0.5:
                    result.append(1)
                else:
                    result.append(0)

            cnt = 0
            for prediction, ground_truth in zip(result, y_validation):
                if prediction == ground_truth:
                    cnt += 1
            val_acc = cnt / len(result)

            print('Epoch %d ---- Training loss: %f ---- Validation loss: %f ---- Training acc: %f ---- Validation acc: %f'
                  % (i, loss.item(), val_loss.item(), acc, val_acc))

        torch.save(model.state_dict(), 'models/dnn.plk')

    def test(self):
        dtype = torch.float
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, pid = process.merge_data('test', if_one_hot=False, continuous=True)
        x = torch.tensor(feature, dtype=dtype)

        model = DNN()
        model.load_state_dict(torch.load('models/dnn.plk'))
        pre = model(x)

        probability = [[i.item()] for i in pre.data]
        scaler = MinMaxScaler()
        probability = scaler.fit_transform(probability)

        probability = pd.Series([i[0] for i in probability], name='dnn_probability').to_frame()
        probability.to_csv(path_or_buf='probability/dnn_probability.csv', index=False)

        result = []

        for pre_i in pre.data:
            if pre_i.item() > 0.5:
                result.append(1)
            else:
                result.append(0)

        prediction = pd.Series((i for i in result), name='Survived')

        submission = pd.concat([pid, prediction], axis=1)

        submission.to_csv(path_or_buf='data/gender_submission.csv', index=False)


if __name__ == '__main__':
    ti = Titanic()
    # ti.train()
    ti.test()
    # modify_csv('prediction_dnn.csv')
