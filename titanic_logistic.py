import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = pd.read_csv(path)
    return data


class Titanic:

    def __init__(self):
        self.model = None
        self.age_scaler = StandardScaler()
        self.fare_scaler = StandardScaler()
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
            feature[:, 2] = np.reshape(self.age_scaler.transform(np.reshape(feature[:, 2], (-1, 1))), (-1))
            feature[:, 5] = np.reshape(self.age_scaler.transform(np.reshape(feature[:, 5], (-1, 1))), (-1))
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        data = load_data('data/train.csv')
        feature, label = self.data_pre_process(data, 'train')
        self.model = LogisticRegression(
            solver='liblinear',
            max_iter=100, multi_class='ovr',
            verbose=1
        ).fit(feature, label)
        acc = self.model.score(feature, label)
        print(acc)

    def test(self):
        data = load_data('data/test.csv')
        feature, p_id = self.data_pre_process(data)
        pre_id = p_id.reset_index(drop=True)
        if self.model is not None:
            pre = self.model.predict(feature)
            prediction = pd.Series(data=pre, name='Survived').to_frame()
            result = pre_id.to_frame().join(prediction)
            result.to_csv(path_or_buf='prediction_logistic.csv', index=False)
            return result
        else:
            print('Model not exists.')
            return


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
