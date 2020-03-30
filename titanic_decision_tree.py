import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from data_process import PreProcess


class Titanic(object):

    def __init__(self):
        self.clf = DecisionTreeClassifier()
        pass

    def train(self):
        process = PreProcess()
        process.load_data('data/train.csv')
        feature, label = process.merge_data(mode='train', if_one_hot=False, continuous=False)

        self.clf.fit(feature, label)

        acc = round(self.clf.score(feature, label) * 100, 2)

        print('Training acc: %f' % acc)

    def test(self):
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, pid = process.merge_data('test', if_one_hot=False, continuous=False)

        pre = self.clf.predict(feature)
        probability = self.clf.predict_proba(feature)

        probability = pd.Series([i[0] for i in probability], name='DT_probability').to_frame()
        probability.to_csv(path_or_buf='probability/DT_probability.csv', index=False)

        prediction = pd.Series((i for i in pre), name='Survived')
        submission = pd.concat([pid, prediction], axis=1)
        submission.to_csv(path_or_buf='data/DT_submission.csv', index=False)


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
