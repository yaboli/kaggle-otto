import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import time


def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        # manually set 'num_class'
        xgb_params = alg.get_xgb_params()
        xgb_params['num_class'] = 9
        xgTrain = xgb.DMatrix(X_train, label=y_train)
        print("\nStarting cross-validation")
        cvresult = xgb.cv(xgb_params,
                          xgTrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='mlogloss',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0,
                          callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                                     xgb.callback.early_stop(3)]
                          )
        print("Cross-validation has completed")
        print("\nPrinting cv result")
        print(cvresult)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm
    print("\nStart fitting the algorithm on the training data:")
    alg.fit(X_train, y_train, eval_metric='mlogloss')
    print("Model fitting has completed")

    # Predict training set
    y_pred = alg.predict(X_train)
    y_pred_prob = alg.predict_proba(X_train)

    # Print model report
    print("\n-----------Model Report-----------")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred))
    print("Log Loss Score (Train): %f" % metrics.log_loss(y_train, y_pred_prob))

    # Plot feature importance figure
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

# read training data
train = pd.read_csv('../train.csv')
# drop 'id' column
train.drop('id', axis=1, inplace=True)
# extract data in the 'target' column and convert to numeric
y_train = train['target'].values
le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
# drop 'target' column
train.drop('target', axis=1, inplace=True)
X_train = train.values

# initialize XGBClassifier()
xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective='multi:softprob',
    seed=27)

# start timer
start = time.time()

# start fitting the model
modelfit(xgb1, X_train, y_train)

# end timer
end = time.time()
print("\nTotal time : {:.2f} {}".format((end - start) / 60, "minutes"))
