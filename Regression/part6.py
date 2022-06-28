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


# updated preprocess phase:
test_data, train_data, raw_blood = pre.updated_preprocess()

train_data = train_data.copy()

y = train_data['covid_score']
X = train_data.drop(['covid_score', 'spread_score'] ,axis = 1)

# alphas = np.logspace(start = -3, stop = 3 , base = 10, num = 50 )
alphas = np.arange(3,7,0.1)
poly_train_scores = []
poly_val_scores = []
linear_train_scores = []
linear_val_scores = []

for alpha in alphas:
    poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', MinMaxScaler()),('Ridge', Ridge(alpha=alpha, fit_intercept=True))])
    scores = cross_validate(poly_reg, X , y, return_train_score=True,scoring='neg_mean_squared_error')
    poly_train_scores.append(np.mean(scores['train_score']))
    poly_val_scores.append(np.mean(scores['test_score']))

    ridge = Ridge(alpha = alpha,fit_intercept=True)
    scores = cross_validate(ridge, X , y, return_train_score=True,scoring='neg_mean_squared_error')
    linear_train_scores.append(np.mean(scores['train_score']))
    linear_val_scores.append(np.mean(scores['test_score']))

    
best_train_score = max(poly_train_scores)
train_opt_alpha = alphas[np.argmax(poly_train_scores)]

best_val_score = max(poly_val_scores)
val_opt_alpha = alphas[np.argmax(poly_val_scores)]

print ('opt train score is: ', best_train_score, 'with alpha ', train_opt_alpha, '\n')
print ('opt val score is: ', best_val_score, 'with alpha ', val_opt_alpha, '\n')

plt.semilogx(alphas, poly_train_scores, label = 'poly train scores')
plt.semilogx(alphas, poly_val_scores, label = 'poly val scores')
plt.semilogx(alphas, linear_train_scores, label = 'linear train scores')
plt.semilogx(alphas, linear_val_scores, label = 'linear val scores')

plt.xlabel('Alphas', fontsize = 20)
plt.ylabel('neg mean squared error', fontsize = 20)
plt.title('score of poly ridge regressor w.r.t. alpha parameter', fontsize = 32)
plt.legend(fontsize = 20)
plt.show()

opt_poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', MinMaxScaler()),('Ridge', Ridge(alpha=val_opt_alpha, fit_intercept=True))])
predictions = opt_poly_reg.fit(X, y).predict(X)


