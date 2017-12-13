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
    max_depth=9,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    seed=27)

# set grid search parameters
subsample = [i/10.0 for i in range(6, 10)]
colsample_bytree = [i/10.0 for i in range(6, 10)]
num_fits = len(subsample)*len(colsample_bytree)*5
param_grid = dict(subsample=subsample,
                  colsample_bytree=colsample_bytree
                  )

kfold = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(model,
                           param_grid,
                           scoring="neg_log_loss",
                           cv=kfold,
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
print("parameters: {}".format(model.get_xgb_params()))
print("\nTotal time : {:.2f} {}".format((end - start) / 60, "minutes"))
