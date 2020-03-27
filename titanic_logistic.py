import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_process import PreProcess


class Titanic:

    def __init__(self):
        self.model = None
        pass

    def train(self):
        process = PreProcess()
        process.load_data('data/train.csv')
        feature, label = process.merge_data(if_one_hot=False, continuous=True)
        self.model = LogisticRegression(
            solver='liblinear',
            max_iter=10000, multi_class='ovr',
            verbose=1
        ).fit(feature, label)
        acc = self.model.score(feature, label)
        print(acc)

    def test(self):
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, p_id = process.merge_data('test', if_one_hot=False, continuous=True)
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
