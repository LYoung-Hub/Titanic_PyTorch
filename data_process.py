import pandas as pd


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

    def process_label(self):
        return pd.get_dummies(self.data['Survived'])

    def process_Pclass(self):
        return pd.get_dummies(self.data['Pclass'])

    def process_embarked(self):
        return pd.get_dummies(self.data['Embarked'])

    def get_pid(self):
        return self.data['PassengerId']

    def merge_data(self):
        sex = self.process_sex()
        Pclass = self.process_Pclass()
        embarked = self.process_embarked()
        label = self.process_label()

        feature = pd.concat([sex, Pclass, embarked], axis=1, sort=False)


# process = PreProcess()
# process.load_data(path='./data/train.csv')
# process.merge_data()
