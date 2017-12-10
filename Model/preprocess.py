import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path_train = 'D:/kaggle-otto/train.csv'
# file_path_test = 'D:/kaggle-otto/test.csv'


def preprocess():
    print('\nPreprocessing data...')
    # load training data
    data = pd.read_csv(file_path_train)
    # drop 'id' column from training set
    data.drop('id', axis=1, inplace=True)
    # extract labels from 'target' column and do one-hot encode
    labels = data['target'].values
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # drop 'target' column from training set
    data.drop('target', axis=1, inplace=True)
    inputs = data.values
    print('Preprocessing completed\n')
    return inputs, labels
