import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import time


start = time.time()

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

# initialize xgboost model
model = XGBClassifier(
    learning_rate=0.1,
    # n_estimators=100,
    n_estimators=615,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective='multi:softprob',
    seed=27)

# set grid search parameters
reg_alpha = [1e-5, 1e-2, 0.1, 1]
# reg_alpha = [1e-6, 1e-5, 2e-5, 5e-5]
# reg_lambda = [1e-5, 1e-2, 0.1, 1]
# reg_lambda = [1e-3, 1e-02, 2e-02, 5e-02]
param_grid = dict(
                    reg_alpha=reg_alpha,
                    # reg_lambda=reg_lambda
                  )

kfold = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(model,
                           param_grid,
                           scoring="neg_log_loss",
                           cv=kfold,
                           verbose=100)

grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

end = time.time()
print("parameters: {}".format(model.get_xgb_params()))
print("\nTotal time : {:.2f} {}".format((end - start) / 60, "minutes"))
