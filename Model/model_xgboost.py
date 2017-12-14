import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
import pickle
import time
from Model.preprocess import preprocess

# start timer
start = time.time()

inputs, labels = preprocess()

# initialize xgb classifier
model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.9,
    reg_lambda=0.01,
    objective='multi:softprob'
)

# split data into train data and validation data (for purpose of quicker observation)
test_size = 0.3
X_train, X_cv, y_train, y_cv = train_test_split(inputs, labels, test_size=test_size)
# fit model with partial training set
print('\nFitting model with training set...')
model.fit(X_train, y_train, eval_metric='mlogloss')
print('Fitting completed')
# save trained model
print('\nSaving model to file...')
pickle.dump(model, open("otto_xgb.pickle.dat", "wb"))
print('Model saved')

y_pred_train = model.predict(X_train)
y_pred_prob_train = model.predict_proba(X_train)
y_pred_cv = model.predict(X_cv)
y_pred_prob_cv = model.predict_proba(X_cv)

# print results
print("\n------------- Model Report -------------")
print("Accuracy (train): %.6g" % metrics.accuracy_score(y_train, y_pred_train))
print("Log Loss Score (train): %f" % metrics.log_loss(y_train, y_pred_prob_train))
print("Accuracy (validation): %.6g" % metrics.accuracy_score(y_cv, y_pred_cv))
print("Log Loss Score (validation): %f" % metrics.log_loss(y_cv, y_pred_prob_cv))

# # train with selected features
# thresholds = ['mean', '0.75*mean', '0.66*mean', '0.5*mean', '0.33*mean',  '0.2*mean', '0.1*mean']
# for threshold in thresholds:
#     selection = SelectFromModel(model, threshold=threshold, prefit=True)
#     select_train_data = selection.transform(X_train)
#     selection_model = xgb.XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=5,
#         min_child_weight=1,
#         gamma=0.2,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=1,
#         objective='multi:softprob'
#     )
#     selection_model.fit(select_train_data, y_train)
#     select_test_data = selection.transform(X_cv)
#     y_pred_cv = selection_model.predict(select_test_data)
#     y_pred_prob_cv = selection_model.predict_proba(select_test_data)
#     num_of_features = select_train_data.shape[1]
#     print("\n-------------Model Report ({:d} features, {})-------------".format(num_of_features, threshold))
#     print("Accuracy : %.4g" % metrics.accuracy_score(y_cv, y_pred_cv))
#     print("Log Loss Score (Train): %f" % metrics.log_loss(y_cv, y_pred_prob_cv))

# stop timer
end = time.time()
print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))
