import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from scipy import interp
from scipy.stats import ks_2samp
import time

path = '../'
fname_train = 'train.csv'
fname_test = 'test.csv'

# start timer
start = time.time()

print('\nPreprocessing training set...')
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
print('Preprocessing completed')

# initialize xgb classifier
model = xgb.XGBClassifier(
    learning_rate=0.1,
    # n_estimators=100,
    n_estimators=615,
    max_depth=5,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective='multi:softprob',
    # reg_alpha=1,
    # reg_lambda=0.05
)

# # initialize with huaqiang's parameters
# model = xgb.XGBClassifier(
#     learning_rate=0.05,
#     max_depth=10,
#     gamma=5,
#     subsample=0.7,
#     colsample_bytree=0.7,
#     objective='multi:softprob'
# )

# ------------------------PREDICT WITH SPLIT TRAINING SET---------------------------------
# split training set into train set and test set (for purpose of quicker observation)
test_size = 0.1
train_data, test_data, train_predictions, test_predictions = train_test_split(X_train,
                                                                              y_train,
                                                                              test_size=test_size)
# Binarize predictions
y_b = label_binarize(test_predictions, classes=range(0, 9))
n_classes = y_b.shape[1]

# fit model with partial training set
print('\nFitting model with training set...')
model.fit(train_data, train_predictions, eval_metric='mlogloss')
print('Fitting completed')

# evaluate predictions
print('\nPredicting with trained model...')
y_pred = model.predict(test_data)
y_pred_prob = model.predict_proba(test_data)
print('Prediction completed')
print("\n-------------Model Report (on split training set, 9:1)-------------")
print("Accuracy : %.4g" % metrics.accuracy_score(test_predictions, y_pred))
print("Log Loss Score : %f" % metrics.log_loss(test_predictions, y_pred_prob))
print("Parameters: {}".format(model.get_xgb_params()))

# perform Kolmogorov-Smirnov test
# group test data by classes
dict_ks = dict()
for i in range(0, len(test_predictions)):
    label_index = test_predictions[i]
    label = str(label_index)
    if label in dict_ks:
        dict_ks[label].append(test_data[i])
    else:
        dict_ks[label] = [test_data[i]]

# compute accuracy for each class
array_pg = []
array_pb = []
diff = []
for j in range(0, 9):
    label = str(j)
    y = model.predict(dict_ks[label])
    pred = np.full(len(dict_ks[label]), j)
    Pg = metrics.accuracy_score(pred, y)
    array_pg.append(Pg)
    Pb = 1 - Pg
    array_pb.append(Pb)
    diff.append(abs(Pg - Pb))

ks_result = ks_2samp(array_pg, array_pb)
print('Kolmogorov-Smirnov test: statistic = {}, pvalue = {}; Max|F1 - F2|: {:.4f}'.format(ks_result.statistic, ks_result.pvalue, np.amax(diff)))

# compute PSI
# training ratio
Rg_t = metrics.accuracy_score(train_predictions, model.predict(train_data))
Rb_t = 1 - Rg_t
# validation ratio
Rg_v = metrics.accuracy_score(test_predictions, y_pred)
Rb_v = 1 - Rg_v
PSI = (Rg_t - Rg_v)*np.log(Rg_t/Rg_v) + (Rb_t - Rb_v)*np.log(Rb_t/Rb_v)
print('Population stability index: {:.4f}'.format(PSI))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_b[:, i], y_pred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_b.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# # Plot of a ROC curve for a specific class
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# Plot ROC curves for the multiclass problem
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
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'b', 'g',
                'r', 'c', 'm', 'y'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics of all classes')
plt.legend(loc="lower right")
plt.show()

# # train with selected features
# thresholds = ['mean', '0.75*mean', '0.66*mean', '0.5*mean', '0.33*mean',  '0.2*mean', '0.1*mean']
# for threshold in thresholds:
#     selection = SelectFromModel(model, threshold=threshold, prefit=True)
#     select_train_data = selection.transform(train_data)
#     selection_model = xgb.XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=5,
#         min_child_weight=1,
#         gamma=0.2,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=1,
#         objective='multi:softprob',
#         seed=27
#     )
#     selection_model.fit(select_train_data, train_predictions)
#     select_test_data = selection.transform(test_data)
#     y_pred = selection_model.predict(select_test_data)
#     y_pred_prob = selection_model.predict_proba(select_test_data)
#     num_of_features = select_train_data.shape[1]
#     print("\n-------------Model Report ({:d} features, {})-------------".format(num_of_features, threshold))
#     print("Accuracy : %.4g" % metrics.accuracy_score(test_predictions, y_pred))
#     print("Log Loss Score (Train): %f" % metrics.log_loss(test_predictions, y_pred_prob))
# ------------------------PREDICT WITH SPLIT TRAINING SET END---------------------------------

# # ------------------------PREDICT WITH TEST SET---------------------------------
# # load test set
# print('\nLoading test set...')
# X_test = np.loadtxt(path + fname_test, delimiter=',', skiprows=1, usecols=range(1, 94))
# print('Test set loaded')
# # fit model with entire training set
# print('\nFitting model with training set...')
# model.fit(X_train, y_train, eval_metric='mlogloss')
# print('Fitting completed')
# # make predictions for test set
# print('\nPredicting with trained model...')
# y_pred_prob = model.predict_proba(X_test)
# print('Prediction completed')
#
# # print out predicted probabilities on test set
# print('\nPredicted probabilities on test set:')
# print(y_pred_prob)
#
# # tabulate predictions
# pred_prob_matrix = np.zeros((144368, 10), dtype=np.float)
# for i in range(0, 144368):
#     pred_prob_matrix[i][0] = i + 1
#     for j in range(0, 9):
#         col = j + 1
#         pred_prob_matrix[i][col] = y_pred_prob[i][j]
#
# print('\nPredicted probabilities (tabulated):')
# print(pred_prob_matrix)
#
# # save predictions into .csv file
# fname_submission = 'kaggle-otto-submission.csv'
# with open(fname_submission, 'wb') as file:
#     file.write(b'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
#     np.savetxt(file, pred_prob_matrix, fmt='%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f', delimiter=',')
# # ------------------------PREDICT WITH TEST SET END---------------------------------

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))