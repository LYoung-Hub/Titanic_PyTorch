import pandas as pd
from os import listdir
from os.path import isfile, join


class MergeResults(object):

    def __init__(self):
        # read id
        self.test_path = './data/test.csv'
        self.test_data = pd.read_csv(self.test_path)

        # read probability
        root = './probability/'
        file_list = [f for f in listdir(root) if isfile(join(root, f))]
        data_list = []
        for file_name in file_list:
            file_path = root + file_name
            data_list.append(pd.read_csv(file_path))

        self.data = pd.concat(data_list, axis=1)

    def print_statistics(self):
        # statistic:)
        print('dnn std: %f' % self.data['dnn_probability'].std())
        print('svm std: %f' % self.data['svm_probability'].std())
        print('logistic std: %f' % self.data['logistic_probability'].std())
        print('GBDT std: %f' % self.data['GBDT_probability'].std())
        print('xgboost std: %f' % self.data['XGBoost_probability'].std())
        print('RF std: %f' % self.data['RF_probability'].std())
        print('KNN std: %f' % self.data['knn_probability'].std())
        print('DT std: %f' % self.data['DT_probability'].std())

        print('dnn mean: %f' % self.data['dnn_probability'].mean())
        print('svm mean: %f' % self.data['svm_probability'].mean())
        print('logistic mean: %f' % self.data['logistic_probability'].mean())
        print('GBDT mean: %f' % self.data['GBDT_probability'].mean())
        print('xgboost mean: %f' % self.data['XGBoost_probability'].mean())
        print('RF mean: %f' % self.data['RF_probability'].mean())
        print('KNN mean: %f' % self.data['knn_probability'].mean())
        print('DT std: %f' % self.data['DT_probability'].std())

    def merge_logic(self):
        result = []
        for i in range(len(self.data)):
            if self.data['dnn_probability'][i] > 0.9:
                result.append(1)
                continue

            # if self.data['GBDT_probability'][i] > 0.8:
            #     result.append(1)
            #     continue

            if (self.data['RF_probability'][i] > 0.6 and self.data['GBDT_probability'][i] > 0.6) or\
                    (self.data['svm_probability'][i] > 0.7 and self.data['GBDT_probability'][i] > 0.6) or\
                    (self.data['RF_probability'][i] > 0.6 and self.data['svm_probability'][i] > 0.7):
                result.append(1)
                continue

            average = 0.1 * self.data['GBDT_probability'][i] + 0.1 * self.data['logistic_probability'][i]\
                      + 0.1 * self.data['RF_probability'][i] + 0.1 * self.data['XGBoost_probability'][i]\
                      + 0.3 * self.data['svm_probability'][i] + 0.1 * self.data['knn_probability'][i]\
                      + 0.1 * self.data['DT_probability'][i] + 0.1 * self.data['dnn_probability'][i]

            if average > 0.5:
                result.append(1)
            else:
                result.append(0)

        submission = pd.Series(data=result, name='Survived')

        result = pd.concat([self.test_data['PassengerId'], submission], axis=1)
        result.to_csv('./data/ensemble_submission_logic.csv', index=False)

    def merge_vote(self):
        votes = pd.concat([(self.data[i] > 0.5) * 1 for i in self.data], axis=1)
        scores = (votes.sum(axis=1) >= 4) * 1

        submission = pd.Series(scores, name='Survived')

        result = pd.concat([self.test_data['PassengerId'], submission], axis=1)
        result.to_csv('./data/ensemble_submission_vote.csv', index=False)

    def merge_mean(self):
        scores = self.data.mean(axis=1)
        scores = [int(round(i)) for i in scores]

        submission = pd.Series(scores, name='Survived')

        result = pd.concat([self.test_data['PassengerId'], submission], axis=1)
        result.to_csv('./data/ensemble_submission_mean.csv', index=False)


if __name__ == '__main__':
    merge = MergeResults()
    merge.print_statistics()
    merge.merge_logic()
    merge.merge_vote()
    merge.merge_mean()
