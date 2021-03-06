import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from data_process import PreProcess
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = pd.read_csv(path)
    return data

def ROC_curve(y, prediction):
    prediction = prediction.reshape(1, -1)
    y = y.values.reshape(-1, 1)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(prediction)[::-1][0]
    index = prediction.argsort()[::-1][0]
    y_sort = y[index]
#     print(y_sort)
    tpr = []
    fpr = []
    thr = []

    for i,item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    return tpr, fpr

class Titanic:
    def __init__(self):
        self.model = None
        self.age_scaler = StandardScaler()
        self.fare_scaler = StandardScaler()
        # pass

    def data_pre_process(self, data, mode='test'):
        # variables selection, normalization
        data['Sex'] = data['Sex'].replace(['male', 'female'], [1, 0])

        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Pclass'] = data['Pclass'].fillna(-1)
        data['SibSp'] = data['SibSp'].fillna(-1)
        data['Parch'] = data['Parch'].fillna(-1)

        bin_count = 5
        data['Age_bin'] = pd.cut(data['Age'], bin_count, labels=False)
        data['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'], [0, 1, 2])
        data['Embarked'] = data['Embarked'].fillna(3)
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Fare_bin'] = pd.cut(data['Fare'], bin_count, labels=False)
        # data['Cabin'] = data['Cabin'].fillna('None')

        feature = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Fare_bin', 'Age_bin']].values
        if mode == 'train':
            label = data['Survived']
            return feature, label
        else:
            p_id = data['PassengerId']
            return feature, p_id

    def train(self):
        # process = PreProcess()
        # process.load_data('data/train.csv')
        # feature, label = process.merge_data('train', if_one_hot=False)
        data = pd.read_csv('./data/train.csv')
        feature, label = self.data_pre_process(data, mode='train')

        param = {
            'booster': 'gbtree',
            'base_score': 0.5,
            'colsample_bylevel': 1,
            'colsample_bytree': 1,
            'gamma': 0,
            'learning_rate': 0.1,
            'max_delta_step': 0,
            'max_depth': 2,
            'min_child_weight': 1,
            'missing': None,
            'n_estimators': 150,
            'n_jobs': 1,
            'objective': 'binary:logistic',
            'random_state': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'seed': 1,
            'silent': True,
            'subsample': 1
        }

        data_train = xgb.DMatrix(feature, label=label)

        self.XGBmodel = xgb.train(param, data_train, num_boost_round=100)

        print("TRAIN")

        predictions = self.XGBmodel.predict(data_train)
        bin_pre = [round(val) for val in predictions]
        acc = np.mean(sum(bin_pre == label) / len(label))
        print("acc = (%.5f)" % acc)
        #
        # train_auc = metrics.roc_auc_score(label, predictions)
        # fpr, tpr = ROC_curve(label, predictions)
        # plt.plot(fpr,tpr)
        # plt.xlabel("FPR")
        # plt.ylabel("TPR")
        # plt.show()


    def test(self):

        # process = PreProcess()
        # process.load_data('data/test.csv')
        # feature, p_id = process.merge_data(mode='test', if_one_hot=False, continuous=False)
        # pre_id = p_id.reset_index(drop=True)
        data = pd.read_csv('./data/test.csv')
        feature, p_id = self.data_pre_process(data, mode='test')
        pre_id = p_id.reset_index(drop=True)

        feature_in = xgb.DMatrix(feature)

        if self.XGBmodel is not None:
            # pre = [int(round(value)) for value in self.XGBmodel.predict(feature_in)]

            pre = self.XGBmodel.predict(feature_in)
            prediction = pd.Series(data=pre, name='XGBoost_probability').to_frame()
            print(pre)
            result = prediction
            result.to_csv(path_or_buf=('./probability/XGBoost_probability.csv'), index=False)
            return result

            # prediction = pd.Series(data=pre, name='Survived').to_frame()
            # print(pre)
            # result = pre_id.to_frame().join(prediction)
            # result.to_csv(path_or_buf='prediction_xgboost.csv', index=False)
            # return result
        else:
            print('Model not exists.')
            return


if __name__ == '__main__':
    ti = Titanic()
    ti.train()
    ti.test()
