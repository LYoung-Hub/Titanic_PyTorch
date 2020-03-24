import pandas as pd

class preprocess():

    def __init__(self):
        self.path = ""
        self.data = None

    def load_data(self):
        path = './data/train.csv'
        data = pd.read_csv(path)

    def process_sex(self):
        self.data['Sex'] = self.data['Sex'].replace(['male', 'female'], [1, 0])





def data_pre_process(self, data, mode='test'):
    # variables selection, normalization

    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Pclass_bin1'] = (data['Pclass'] == 1).replace([True, False], [1, 0])
    # data['Pclass_bin2']

    # onehot to be
    data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
    data['Embarked'] = data['Embarked'].fillna(3)

    # onehot to be
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