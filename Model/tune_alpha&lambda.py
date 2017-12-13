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
    subsample=0.7,
    colsample_bytree=0.9,
    objective='multi:softprob',
    seed=27)

# set grid search parameters
# reg_alpha = [1e-5, 1e-2, 0.1, 1]
reg_lambda = [1e-5, 1e-2, 0.1, 1]
# num_fits = len(reg_alpha)*5
num_fits = len(reg_lambda)*5
param_grid = dict(
                  # reg_alpha=reg_alpha,
                  reg_lambda=reg_lambda
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
