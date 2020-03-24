import pandas as pd
import numpy as np


class PreProcess(object):

    def __init__(self):
        self.path = ""
        self.data = None

    def load_data(self, path):
        self.data = pd.read_csv(path)

    def process_sex(self):
        return pd.get_dummies(self.data['Sex'])

    def process_age(self):
        bins = [0, 4, 10, 15, 100]
        age_data = self.data['Age'].fillna(self.data['Age'].median())
        age_bins = pd.cut(age_data, bins, labels=[0, 1, 2, 3])
        return pd.get_dummies(age_bins)

    def process_SibSp(self):
        bins = [-5, -1, 0, 1, 100]
        sib_data = self.data['SibSp'].fillna(-1)
        sib_bins = pd.cut(sib_data, bins, labels=[0, 1, 2, 3])
        return pd.get_dummies(sib_bins)

    def process_parch(self):
        bins = [-5, -1, 0, 1, 100]
        patch_data = self.data['Parch'].fillna(-1)
        patch_bins = pd.cut(patch_data, bins, labels=[0, 1, 2, 3])
        return pd.get_dummies(patch_bins)

    # def process_label(self):
    #     return pd.get_dummies(self.data['Survived'])

    def process_Pclass(self):
        return pd.get_dummies(self.data['Pclass'])

    def process_fare(self):
        fare_bins = pd.qcut(self.data['Fare'], 4, labels=[0, 1, 2, 3])
        return pd.get_dummies(fare_bins)

    def process_embarked(self):
        return pd.get_dummies(self.data['Embarked'])

    def get_pid(self):
        return self.data['PassengerId']

    def merge_data(self, mode='train'):
        sex = self.process_sex()
        Pclass = self.process_Pclass()
        embarked = self.process_embarked()
        age = self.process_age()
        sibsp = self.process_SibSp()
        parch = self.process_parch()

        feature = pd.concat([sex, Pclass, embarked, age, sibsp, parch], axis=1, sort=False)

        if mode == 'train':
            label = self.data['Survived']
            return feature.to_numpy(), np.reshape(label.to_numpy(), (-1, 1))

        pid = self.get_pid()
        return feature.to_numpy(), pid


# process = PreProcess()
# process.load_data(path='./data/train.csv')
# process.merge_data()
