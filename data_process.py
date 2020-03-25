import pandas as pd
import numpy as np


class PreProcess(object):

    def __init__(self):
        self.path = ""
        self.data = None

    def load_data(self, path):
        self.data = pd.read_csv(path)

    def process_sex(self, if_one_hot=True):
        if if_one_hot:
            return pd.get_dummies(self.data['Sex'])
        else:
            return self.data['Sex'].replace(['male', 'female'], [0, 1]).to_frame()
            # return self.data['Sex'].to_frame()

    def process_age(self, if_one_hot=True):
        bins = [0, 4, 10, 15, 100]
        age_data = self.data['Age'].fillna(self.data['Age'].median())
        age_bins = pd.cut(age_data, bins, labels=[0, 1, 2, 3])
        if if_one_hot:
            return pd.get_dummies(age_bins)
        else:
            return age_bins.to_frame()

    def process_SibSp(self, if_one_hot=True):
        bins = [-5, -1, 0, 1, 100]
        sib_data = self.data['SibSp'].fillna(-1)
        sib_bins = pd.cut(sib_data, bins, labels=[0, 1, 2, 3])
        if if_one_hot:
            return pd.get_dummies(sib_bins)
        else:
            return sib_bins.to_frame()

    def process_parch(self, if_one_hot=True):
        bins = [-5, -1, 0, 1, 100]
        patch_data = self.data['Parch'].fillna(-1)
        patch_bins = pd.cut(patch_data, bins, labels=[0, 1, 2, 3])
        if if_one_hot:
            return pd.get_dummies(patch_bins)
        else:
            return patch_bins.to_frame()

    def process_Pclass(self, if_one_hot=True):
        if if_one_hot:
            return pd.get_dummies(self.data['Pclass'])
        else:
            return self.data['Pclass'].to_frame()

    def process_fare(self, if_one_hot=True):
        fare_bins = pd.qcut(self.data['Fare'], 4, labels=[0, 1, 2, 3])
        if if_one_hot:
            return pd.get_dummies(fare_bins)
        else:
            return fare_bins.to_frame()

    def process_embarked(self, if_one_hot=True):
        if if_one_hot:
            return pd.get_dummies(self.data['Embarked'])
        else:
            return self.data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2]).to_frame()

    def get_pid(self):
        return self.data['PassengerId']

    def merge_data(self, mode='train', if_one_hot=True):
        sex = self.process_sex(if_one_hot)
        Pclass = self.process_Pclass(if_one_hot)
        embarked = self.process_embarked(if_one_hot)
        age = self.process_age(if_one_hot)
        sibsp = self.process_SibSp(if_one_hot)
        parch = self.process_parch(if_one_hot)
        fare = self.process_fare(if_one_hot)

        feature = pd.concat([sex, Pclass, embarked, age, sibsp, parch, fare], axis=1, sort=False)

        if mode == 'train':
            label = self.data['Survived']
            return feature.to_numpy(), np.reshape(label.to_numpy(), (-1, 1))

        pid = self.get_pid()
        return feature.to_numpy(), pid


# process = PreProcess()
# process.load_data(path='./data/train.csv')
# process.merge_data()
