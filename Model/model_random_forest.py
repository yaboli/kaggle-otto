import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

path = '../'
fname_train = 'train.csv'
fname_test = 'test.csv'

# start timer
start = time.time()

# load training set
train = pd.read_csv(path + fname_train)
# drop 'id' column from training set
train.drop('id', axis=1, inplace=True)
# extract data in the 'target' column and convert them to numeric
y_train = train['target'].values
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
# drop 'target' column from training set
train.drop('target', axis=1, inplace=True)
X_train = train.values

# initialize random forest classifier
model = RandomForestClassifier(n_estimators=200,
                               max_depth=5,
                               # random_state=27
                               )

# ------------------------PREDICT WITH SPLIT TRAINING SET---------------------------------
# split training set into train set and test set (for purpose of quicker observation)
seed = 7
test_size = 0.33
train_data, test_data, train_predictions, test_predictions = train_test_split(X_train, y_train,
                                                                              test_size=test_size, random_state=seed)

# fit model with 2/3 of training set
model.fit(train_data, train_predictions)

# evaluate predictions
y_pred = model.predict(test_data)
y_pred_prob = model.predict_proba(test_data)
print("\n-------------Model Report (Random Forest)-------------")
print("Accuracy : %.4g" % metrics.accuracy_score(test_predictions, y_pred))
print("Log Loss Score (Train): %f" % metrics.log_loss(test_predictions, y_pred_prob))
# ------------------------PREDICT WITH SPLIT TRAINING SET END---------------------------------

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))