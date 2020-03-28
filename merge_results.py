import pandas as pd
from os import listdir
from os.path import isfile, join

root = './probability/'
file_list = [f for f in listdir(root) if isfile(join(root, f))]
data_list = []
for file_name in file_list:
    file_path = root + file_name
    data_list.append(pd.read_csv(file_path))

data = pd.concat(data_list, axis=1)

# read id
test_path = './data/test.csv'
test_data = pd.read_csv(test_path)
scores = data.mean(axis=1)
scores = [int(round(i)) for i in scores]
scores = pd.Series(scores, name='Survived')

result = pd.concat([test_data['PassengerId'], scores], axis=1)
result.to_csv('./data/ensemble_submission.csv', index=False)

print(result)

