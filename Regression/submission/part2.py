from sklearn.model_selection import train_test_split
import updated_data_loading.preliminary as pre
import HW3_code.verify_gradients as ver_g
import HW3_code.test_lr as tl
import pandas as np
# updated preprocess phase:
test_data, train_data, raw_blood = pre.updated_preprocess()

# A3:
X, y= train_test_split(train_data, test_size=0.2, train_size=0.8, random_state=156)
# functions are implemented in LinearRegressor.py

# A4:
y_train = X[['covid_score']]
X_train = X.copy()
X_train.drop(['covid_score'], axis = 1, inplace = True)
X_train.drop(['spread_score'], axis = 1, inplace = True)

ver_g.compare_gradients(X_train, y_train, deltas=np.logspace(-7, -2, 9))

# A5:
y_val = y[['covid_score']]
X_val = y.copy()
X_val.drop(['covid_score'], axis = 1, inplace = True)
X_val.drop(['spread_score'], axis = 1, inplace = True)
tl.test_lr(X_train, y_train, X_val, y_val, "Train and Validation losses as a function vs. iteration number for different learning rates")