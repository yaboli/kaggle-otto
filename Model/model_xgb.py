import numpy as np
import pandas as pd
import xgboost as xgb
import time

path = '../'
fname_train = 'train.csv'
fname_test = 'test.csv'
start = time.time()

# prepare training data
# load training data
X_training = np.loadtxt(path + fname_train, delimiter=',', skiprows=1, usecols=range(1, 94))
# convert targeted classes to integers
target = pd.read_csv(path+fname_train, index_col=0)['target']
classification_map = pd.Series(range(1, 10), index=target.unique())
y_training = target.map(classification_map).values
# load test data
X_test = np.loadtxt(path + fname_test, delimiter=',', skiprows=1, usecols=range(1, 94))

# fit model with training data
model = xgb.XGBClassifier()
model.fit(X_training, y_training)

# make predictions for test data
y_test = model.predict(X_test)

# print out predictions on test dataset
print('\nPredictions on test dataset:')
print(y_test)

# tabulate predictions
predictions = np.zeros((144368, 10), dtype=np.int32)
for i in range(0, 144368):
    predictions[i][0] = i + 1
    col = y_test[i]
    predictions[i][col] = 1

print('\nPredictions (tabulated):')
print(predictions)

# save predictions into .csv file
fname_submission = 'kaggle-otto-submission.csv'
with open(fname_submission, 'wb') as file:
    file.write(b'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
    np.savetxt(file, predictions.astype(int), fmt='%i', delimiter=',')

end = time.time()
print('\nTime elapsed: ' + str(end - start) + ' seconds')