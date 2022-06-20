from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import updated_data_loading.preliminary as pre
import HW3_code.verify_gradients as ver_g
import pandas as np
# updated preprocess phase:
test_data, train_data, raw_blood = pre.updated_preprocess()

# A3:
X, y= train_test_split(train_data, test_size=0.2, train_size=0.8, random_state=156)
# functions are implemented in LinearRegressor.py

# A4:
X = X.drop(columns = ['covid_score', 'spread_score'])
y = y['covid_score']
ver_g.compare_gradients(X, y, deltas=np.logspace(-7, -2, 9))