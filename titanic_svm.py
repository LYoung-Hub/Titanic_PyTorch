import pandas as pd
from sklearn import svm
from data_process import PreProcess


class Titanic(object):

    def __init__(self):
        self.clf = svm.SVR()
        pass

    def train(self):
        process = PreProcess()
        process.load_data('data/train.csv')
        feature, label = process.merge_data(mode='train', if_one_hot=True, continuous=False)

        self.clf.fit(feature, label)
        pre = self.clf.predict(feature)

        cnt = 0
        for prediction, ground_truth in zip(pre, label):
            if prediction == ground_truth:
                cnt += 1
        acc = cnt / len(pre)

        print('Training acc: %f' % acc)

    def test(self):
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, pid = process.merge_data('test', if_one_hot=True, continuous=False)

        pre = self.clf.predict(feature)

        probability = pd.Series([i for i in pre], name='svm_probability').to_frame()
        probability.to_csv(path_or_buf='probability/svm_probability.csv', index=False)

        # need SVC to get classification result directly
        prediction = pd.Series((i for i in pre), name='Survived')

        submission = pd.concat([pid, prediction], axis=1)

        submission.to_csv(path_or_buf='data/svm_submission.csv', index=False)


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
