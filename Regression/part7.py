# A27:
from sklearn.metrics import mean_squared_error
import updated_data_loading.preliminary as pre
import seaborn as sns
from multiprocessing import dummy
from operator import ne
from tkinter import Y, font
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from plot3d import plot3d
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
import LinearRegressor

test_data, train_data, raw_blood = pre.updated_preprocess()



X_train = train_data.copy()
y_train = X_train[['covid_score']]
X_train.drop(['covid_score'], axis = 1, inplace = True)
X_train.drop(['spread_score'], axis = 1, inplace = True)

X_test = test_data.copy()
y_test = X_test[['covid_score']]
X_test.drop(['covid_score'], axis = 1, inplace = True)
X_test.drop(['spread_score'], axis = 1, inplace = True)

thresh = ((y_test - y_test.mean()) ** 2).sum()


# dummy:
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
y_pred = dummy_regr.predict(X_test)
print("dummy:", mean_squared_error(y_test, y_pred))
# Linear:
"""linear_reggressor = LinearRegressor(0.1)
linear_reggressor.fit(X_train, y_train)
y_pred2 = linear_reggressor.predict(X_test)
print("linear:", mean_squared_error(y_test, y_pred2))"""
# Ridge Linear:
clf = Ridge(alpha=1, fit_intercept=True)
clf.fit(X_train, y_train)
y_pred3 = clf.predict(X_test)
print("Ridge linear:", mean_squared_error(y_test, y_pred3))

# Ridge poly:
clf = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', MinMaxScaler()),('Ridge', Ridge(alpha=5.5, fit_intercept=True))])
clf.fit(X_train, y_train)
y_pred3 = clf.predict(X_test)
print("Ridge poly:", mean_squared_error(y_test, y_pred3))


