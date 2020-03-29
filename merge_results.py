import pandas as pd
from os import listdir
from os.path import isfile, join

# read id
test_path = './data/test.csv'
test_data = pd.read_csv(test_path)

# read probability
root = './probability/'
file_list = [f for f in listdir(root) if isfile(join(root, f))]
data_list = []
for file_name in file_list:
    file_path = root + file_name
    data_list.append(pd.read_csv(file_path))

data = pd.concat(data_list, axis=1)

# statistic
print('dnn std: %f' % data['dnn_probability'].std())
print('svm std: %f' % data['svm_probability'].std())
print('logistic std: %f' % data['logistic_probability'].std())
print('GBDT std: %f' % data['GBDT_probability'].std())
print('xgboost std: %f' % data['XGBoost_probability'].std())
print('RF std: %f' % data['RF_probability'].std())

print('dnn mean: %f' % data['dnn_probability'].mean())
print('svm mean: %f' % data['svm_probability'].mean())
print('logistic mean: %f' % data['logistic_probability'].mean())
print('GBDT mean: %f' % data['GBDT_probability'].mean())
print('xgboost mean: %f' % data['XGBoost_probability'].mean())
print('RF mean: %f' % data['RF_probability'].mean())

result = []
for i in range(len(data)):
    if data['dnn_probability'][i] > 0.9:
        result.append(1)
        continue

    if data['GBDT_probability'][i] > 0.8:
        result.append(1)
        continue

    if data['RF_probability'][i] > 0.6 and data['GBDT_probability'][i] > 0.6:
        result.append(1)
        continue

    average = 0.2 * data['dnn_probability'][i] + 0.15 * data['GBDT_probability'][i] + 0.15 * data['logistic_probability'][i]\
              + 0.2 * data['RF_probability'][i] + 0.05 * data['XGBoost_probability'][i] + 0.25 * data['svm_probability'][i]

    if average > 0.5:
        result.append(1)
    else:
        result.append(0)

submission = pd.Series(data=result, name='Survived')

# read id
test_path = './data/test.csv'
test_data = pd.read_csv(test_path)
scores = data.mean(axis=1)
scores = [int(round(i)) for i in scores]
submission = pd.Series(scores, name='Survived')
# calculate mean value

result = pd.concat([test_data['PassengerId'], submission], axis=1)
result.to_csv('./data/ensemble_submission_2.csv', index=False)
