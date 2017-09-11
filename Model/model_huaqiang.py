#!/usr/bin/python

from __future__ import division

import numpy as np
import matplotlib

matplotlib.use('Agg')
import xgboost as xgb
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import normalize

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def addFea(train_Xn, train_X, num1, num2):
    train_var = tran(train_X[:, 1:], np.var, axis=1)
    train_zero = tran(train_X, np.count_nonzero, axis=1)
    train_zero.shape = 1, -1
    train_zero = map(lambda x: 94 - x, train_zero)
    train_zero = np.transpose(train_zero)
    train_add = np.add(train_X[:, num1], train_X[:, num2])
    train_mul = train_X[:, num1] * train_X[:, num2]
    train_sub = np.subtract(train_X[:, num1], train_X[:, num2])

    with np.errstate(divide='ignore', invalid='ignore'):
        train_div = np.true_divide(train_X[:, 25], train_X[:, 67])
        train_div[train_div == np.inf] = 0
        train_div = np.nan_to_num(train_div)

    train_Xn = np.c_[train_Xn, train_var, train_zero, train_add, train_sub, train_mul, train_div]
    return train_Xn


def tran(array, func, *args, **kwargs):
    res = func(array, *args, **kwargs)
    res.shape = (res.shape[0], 1)
    np.transpose(res)
    return res


def normalizer(lst):
    norm = lst
    for i in range(lst.shape[0]):
        norm[i] = [(float(2 * j) - max(lst[i]) - min(lst[i])) / (max(lst[i]) - min(lst[i])) for j in lst[i]]
        if (i % 500 == 0):
            print(" %.4f done" % (float(i / lst.shape[0])))
    return norm


# label need to be 0 to num_class -1
data = np.loadtxt('../train.csv', skiprows=1, delimiter=',',
                  converters={0: lambda x: int('0'), 94: lambda x: int(x[6:]) - 1})
sz = data.shape

np.random.shuffle(data)
train = data[:int(sz[0] * 0.9), :]
test = data[int(sz[0] * 0.9):, :]

train_X = train[:, :93]
train_Xo = train_X

train_Y = train[:, 94]
train_X = addFea(train_X, train_Xo, 25, 67)
train_X = addFea(train_X, train_Xo, 25, 86)
print(train_X[0, :])

train_X = normalize(train_X, norm='l2')

test_X = test[:, :93]
test_Xo = test_X
test_X = addFea(test_X, test_X, 25, 67)
test_X = addFea(test_X, test_Xo, 25, 86)

test_X = normalize(test_X, norm='l2')

test_Y = test[:, 94]
test_Yb = label_binarize(test_Y, classes=range(0, 9))

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
# param['max_depth'] = int('6')
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 9

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
model = xgb.XGBClassifier(max_depth=10, subsample=0.70, colsample_bytree=0.7, objective='multi:softmax', learning_rate=0.05,
                          gamma=5, silent=0,
                          # n_jobs=40, num_class=9
                          )
model.fit(train_X, train_Y)

# bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
# pred = bst.predict(xg_test)

res = model.predict(test_X)
pred = model.predict_proba(test_X)

# print(model.booster().get_score(importance_type='weight'))
# print(model.booster().plot_importance(model))

'''
for i in range(test_X.shape[0]):
    if res[i] == test_Y[i]:    
        print(pred[i])
        print(test_Y[i])
'''


# pred_b = label_binarize(pred, classes=range(0,9))
def plot(test_Yb, pred, param, filename="otto"):
    fpr = dict()
    tpr = dict()

    roc_auc = dict()

    for i in range(param['num_class']):
        fpr[i], tpr[i], _ = roc_curve(test_Yb[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(test_Yb.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    n_classes = param["num_class"]
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(filename)


error_rate = np.sum(res != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

plt.figure()
xgb.plot_importance(model, max_num_features=20)

plt.savefig("imp.png")

plot(test_Yb, pred, param, "otto")
'''
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
#print(xgb.get_score())
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], param['num_class'])
pred_label = np.argmax(pred_prob, axis=1)
print(bst.get_score())
#print(xgb.plot_importance(bst))
#error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
#print('Test error using softprob = {}'.format(error_rate))
'''