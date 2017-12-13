from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from Model.preprocess import preprocess
import time

X_train, y_train = preprocess()

# initialize xgboost model
model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    seed=27)

# set grid search parameters
max_depth = range(3, 10, 2)  # optimal: 9
min_child_weight = range(1, 6, 2)  # optimal: 1
num_fits = len(max_depth)*len(min_child_weight)*5
param_grid = dict(max_depth=max_depth,
                  min_child_weight=min_child_weight)

kfold = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(model,
                           param_grid,
                           scoring="neg_log_loss",
                           cv=kfold,
                           iid=False,  # not sure about this parameter
                           verbose=num_fits)

start = time.time()
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

end = time.time()
print("\nTotal time : {:.2f} {}".format((end - start) / 60, "minutes"))
