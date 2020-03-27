import pandas as pd
from data_process import PreProcess
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


class Titanic:
    def __init__(self):
        self.GBDT_model = None
        self.RF_model = None
        self.max_acc = 0

    def train(self, model_type):
        process = PreProcess()
        process.load_data('data/train.csv')
        feature, label = process.merge_data(mode='train', if_one_hot=False, continuous=True)

        test_size = 0.3
        lr_list = [0.25, 0.5, 0.8, 1, 1.2]

        for learning_rate in lr_list:
            X_train, X_val, y_train, y_val = train_test_split(feature, label, test_size=test_size, shuffle=True)
            y_train, y_val = y_train.ravel(), y_val.ravel()

            if model_type == 'RF':
                gb_clf = RandomForestClassifier(
                    n_estimators=20,
                    max_features=2,
                    max_depth=2,
                )
            else:
                gb_clf = GradientBoostingClassifier(
                    n_estimators=20,
                    learning_rate=learning_rate,
                    max_features=2,
                    max_depth=2,
                    random_state=0)

            gb_clf.fit(X_train, y_train)
            train_score = gb_clf.score(X_train, y_train)
            valid_score = gb_clf.score(X_val, y_val)
            if valid_score > self.max_acc:
                self.max_acc = valid_score
                if model_type == 'RF':
                    self.RF_model = gb_clf
                else:
                    self.GBDT_model = gb_clf

            print("Learning rate: ", learning_rate)
            print("Accuracy score (training): {0:.3f}".format(train_score))
            print("Accuracy score (validation): {0:.3f}".format(valid_score))
            if model_type == 'RF':
                break

    def test(self, model_type):
        process = PreProcess()
        process.load_data('data/test.csv')
        feature, p_id = process.merge_data(mode='test', if_one_hot=False)
        pre_id = p_id.reset_index(drop=True)

        if model_type == 'RF':
            pre = [int(round(value)) for value in self.RF_model.predict(feature)]
        else:
            pre = [int(round(value)) for value in self.GBDT_model.predict(feature)]

        prediction = pd.Series(data=pre, name='Survived').to_frame()
        print(pre)
        result = pre_id.to_frame().join(prediction)
        result.to_csv(path_or_buf=('prediction_%s.csv' % model_type), index=False)
        return result


if __name__ == '__main__':
    ti = Titanic()
    ti.train('RF')
    ti.test('RF')