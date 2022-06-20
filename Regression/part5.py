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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler




df = pd.read_csv('train_ds.csv')

# plot3d(df = df, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = 'spread_score w.r.t. suger_levels and PCR_05')

X = df[['PCR_05', 'sugar_levels']]
y = df['spread_score']


# alphas = np.logspace(start = -3, stop = 8, base = 10, num = 50 )
alphas = range(50)
linear_train_scores = []
linear_val_scores = []
dummy_train_scores = []
dummy_val_scores = []

for alpha in alphas:
    ridge = Ridge(alpha = alpha,fit_intercept=True)
    scores = cross_validate(ridge, X , y, return_train_score=True,scoring='neg_mean_squared_error')
    linear_train_scores.append(np.mean(scores['train_score']))
    linear_val_scores.append(np.mean(scores['test_score']))
    
    dummy_reg = DummyRegressor()
    dummy_scores = cross_validate(dummy_reg, X , y, return_train_score=True,scoring='neg_mean_squared_error')
    dummy_train_scores.append(np.mean(dummy_scores['train_score']))
    dummy_val_scores.append(np.mean(dummy_scores['test_score']))

best_train_score = max(linear_train_scores)
train_opt_alpha = alphas[np.argmax(linear_train_scores)]

best_val_score = max(linear_val_scores)
val_opt_alpha = alphas[np.argmax(linear_val_scores)]

print ('opt train score is: ', best_train_score, 'with alpha ', train_opt_alpha, '\n')
print ('opt val score is: ', best_val_score, 'with alpha ', val_opt_alpha, '\n')


plt.semilogx(alphas, linear_train_scores, label = 'train scores')
plt.semilogx(alphas, linear_val_scores, label = 'val scores')
plt.semilogx(alphas, dummy_train_scores, label = 'dummy train scores')
plt.semilogx(alphas, dummy_val_scores, label = 'dummy val scores')

plt.xlabel('Alphas', fontsize = 20)
plt.ylabel('neg mean squared error', fontsize = 20)
plt.title("ridge regressor's score w.r.t. alpha", fontsize = 32)
plt.legend(fontsize = 20)
# plt.show()


opt_ridge = Ridge(alpha = val_opt_alpha)
predictions = opt_ridge.fit(X, y).predict(X)

plot3d(df = df, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = "Opt ridge regressor's predictions (blue) vs true labels (red)", predictions = predictions)
#####################################################################################################################################################################################################

# alphas = np.logspace(start = -3, stop = 3 , base = 10, num = 50 )
# alphas = range(50)
poly_train_scores = []
poly_val_scores = []

for alpha in alphas:
    poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', StandardScaler()),('Ridge', Ridge(alpha=alpha, fit_intercept=True))])
    scores = cross_validate(poly_reg, X , y, return_train_score=True,scoring='neg_mean_squared_error')
    poly_train_scores.append(np.mean(scores['train_score']))
    poly_val_scores.append(np.mean(scores['test_score']))
    
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

opt_poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', StandardScaler()),('Ridge', Ridge(alpha=val_opt_alpha, fit_intercept=True))])
predictions = opt_poly_reg.fit(X, y).predict(X)

plot3d(df = df, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = "Opt poly ridge regressor's predictions (blue) vs true labels (red)", predictions = predictions)
