import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from scipy import interp
from scipy.stats import ks_2samp
import pickle
from Model.preprocess import preprocess

fname_test = 'test.csv'

inputs, labels = preprocess()

# apply one-hot encoding to labels
labels_enc = label_binarize(labels, classes=range(0, 9))
n_classes = labels_enc.shape[1]

model = pickle.load(open("D:/kaggle-otto/Model/otto_xgb.pickle.dat", "rb"))

# predict
print('\nPredicting with trained model...')
y_pred_prob = model.predict_proba(inputs)
print('Prediction completed')

# ------------------ ROC ------------------
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(labels_enc[:, i], y_pred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(labels_enc.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area; first aggregate all false positive rates
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
         label='micro-average ROC curve (area = {0:0.6f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.6f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'b', 'g',
#                 'r', 'c', 'm', 'y'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics of all classes')
plt.legend(loc="lower right")
plt.savefig('D:/kaggle-otto/roc/roc_xgb.png')
plt.close()

# # ------------------ Kolmogorov-Smirnov test ------------------
# # group test data by classes
# dict_ks = dict()
# for i in range(0, len(y_cv)):
#     label_index = y_cv[i]
#     label = str(label_index)
#     if label in dict_ks:
#         dict_ks[label].append(X_cv[i])
#     else:
#         dict_ks[label] = [X_cv[i]]
#
# # compute accuracy for each class
# array_pg = []
# array_pb = []
# diff = []
# for j in range(0, 9):
#     label = str(j)
#     y = model.predict(dict_ks[label])
#     pred = np.full(len(dict_ks[label]), j)
#     Pg = metrics.accuracy_score(pred, y)
#     array_pg.append(Pg)
#     Pb = 1 - Pg
#     array_pb.append(Pb)
#     diff.append(abs(Pg - Pb))
#
# ks_result = ks_2samp(array_pg, array_pb) print('Kolmogorov-Smirnov test: statistic = {}, pvalue = {}; Max|F1 - F2|:
#  {:.4f}'.format(ks_result.statistic, ks_result.pvalue, np.amax(diff)))
#
# # ------------------ PSI ------------------
# # training ratio
# Rg_t = metrics.accuracy_score(y_train, model.predict(X_train))
# Rb_t = 1 - Rg_t
# # validation ratio
# Rg_v = metrics.accuracy_score(y_cv, y_pred_cv)
# Rb_v = 1 - Rg_v
# PSI = (Rg_t - Rg_v)*np.log(Rg_t/Rg_v) + (Rb_t - Rb_v)*np.log(Rb_t/Rb_v)
# print('Population stability index: {:.4f}'.format(PSI))

# # ------------------------PREDICT WITH TEST SET---------------------------------
# # load test set
# print('\nLoading test set...')
# X_test = np.loadtxt(path + file_path_test, delimiter=',', skiprows=1, usecols=range(1, 94))
# print('Test set loaded')
# # fit model with entire training set
# print('\nFitting model with training set...')
# model.fit(inputs, labels, eval_metric='mlogloss')
# print('Fitting completed')
# # make predictions for test set
# print('\nPredicting with trained model...')
# y_pred_prob_cv = model.predict_proba(X_test)
# print('Prediction completed')
#
# # print out predicted probabilities on test set
# print('\nPredicted probabilities on test set:')
# print(y_pred_prob_cv)
#
# # tabulate predictions
# pred_prob_matrix = np.zeros((144368, 10), dtype=np.float)
# for i in range(0, 144368):
#     pred_prob_matrix[i][0] = i + 1
#     for j in range(0, 9):
#         col = j + 1
#         pred_prob_matrix[i][col] = y_pred_prob_cv[i][j]
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
