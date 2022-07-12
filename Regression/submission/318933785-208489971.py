import updated_data_loading.preliminary as pre
import seaborn as sns
import matplotlib.pyplot as plt

# updated preprocess phase:
test_data, train_data, raw_blood = pre.updated_preprocess()

# A1:
temp_train = train_data.join(raw_blood)
temp_train.replace({"A+": "A", "A-": "A", "B+": "Others",
                    "B-": "Others", "AB+": "Others", "AB-": "Others", "O+": "O", "O-": "O"}, inplace=True)
rel = sns.kdeplot(data=temp_train, x='covid_score',hue='blood_type', common_norm=False).set(title='Covid Score by Blood Type')

plt.show()


#part 2

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

#part 3 

import imp
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import numpy as np
# A6:
X_train = train_data.copy()
y_train = X_train[['covid_score']]
X_train.drop(['covid_score'], axis = 1, inplace = True)
X_train.drop(['spread_score'], axis = 1, inplace = True)
dummy_scores = []
dummy_train_scores = []

dummy_regr = DummyRegressor(strategy="mean")
scores = cross_validate(dummy_regr, X_train, y_train, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
mean = scores['test_score'].mean()
mean_train = scores['train_score'].mean()
dummy_scores.append(mean)
dummy_train_scores.append(mean_train)

print("dummy_scores (validation):", dummy_scores)
print("dummy_scores (training):", dummy_train_scores)

# A7: cross validation of our Linear regressor
X_train = train_data.copy() #the train_data of entire dataset. 
y_train = X_train[['covid_score']]
X_train.drop(['covid_score'], axis = 1, inplace = True)
X_train.drop(['spread_score'], axis = 1, inplace = True)
linear_validation_scores = []
linear_train_scores = []
lr_list = np.logspace(-9, -1, 9)
max_val = -1000000
lr_opt = 0
for lr in lr_list:
    cur_linear_reggressor = LinearRegressor(lr)
    scores = cross_validate(cur_linear_reggressor, X_train, y_train, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
    mean = scores['test_score'].mean()
    mean_train = scores['train_score'].mean()
    if(mean > max_val):
      max_val = mean
      lr_opt = lr
    linear_validation_scores.append(mean)
    linear_train_scores.append(mean_train)

print("Optimal Validation Error: ",max_val, " and its lr: ", lr_opt)

from matplotlib.pyplot import semilogx
dummy = np.full(lr_list.size, dummy_scores)
plt.semilogx(lr_list, linear_validation_scores, marker = ".", markersize = 15, color = "blue", linewidth=2, label="Validation")
plt.semilogx(lr_list, dummy, color = "green", linewidth=2, label="Dummy Score")
plt.semilogx(lr_list, linear_train_scores, color = "red", linewidth=2, label="Training")
plt.suptitle("Cross-Validated Mean squared error vs. Learning rate values")
plt.legend()
plt.xlabel('Value of Learning rate')
plt.ylabel('Cross-Validated MSE')

plt.show()

#part 4

# A9:
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import cross_validate

X_train = train_data.copy()
y_train = X_train[['covid_score']]
X_train.drop(['covid_score'], axis = 1, inplace = True)
X_train.drop(['spread_score'], axis = 1, inplace = True)
linear_validation_scores = []
linear_train_scores = []
lambda_list = np.logspace(-9, 2, 12)

isFirst = True
max_val = 0
l_opt = 0
for l in lambda_list:
    clf = Ridge(alpha=l, fit_intercept=True)
    scores = cross_validate(clf, X_train, y_train, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
    mean = scores['test_score'].mean()
    mean_train = scores['train_score'].mean()
    if(mean > max_val):
      max_val = mean
      l_opt = l
    if(isFirst == True):
      max_val = mean
      l_opt = l
      isFirst = False
    linear_validation_scores.append(mean)
    linear_train_scores.append(mean_train)

print("Optimal Validation Error: ",max_val, " and its lambda: ", l_opt)

# A10:
# dummy:
from sklearn.dummy import DummyRegressor
dummy_scores = []
dummy_train_scores = []
dummy_regr = DummyRegressor(strategy="mean")
scores = cross_validate(dummy_regr, X_train, y_train, cv=5, return_train_score=True, scoring='neg_mean_squared_error')
mean = scores['test_score'].mean()
mean_train = scores['train_score'].mean()
dummy_scores.append(mean)
dummy_train_scores.append(mean_train)
dummy = np.full(lambda_list.size, dummy_scores)

# plotting:
from matplotlib.pyplot import semilogx
plt.semilogx(lambda_list, linear_validation_scores, marker = ".", markersize = 15, color = "blue", linewidth=2, label="Validation")
plt.semilogx(lambda_list, dummy, color = "green", linewidth=2, label="Dummy")
plt.semilogx(lambda_list, linear_train_scores, color = "red", linewidth=2, label="Training")
plt.suptitle("Cross-Validated Mean squared error vs. Regularization Lambda Values")
plt.legend()
plt.xlabel('Value of Lambda')
plt.ylabel('Cross-Validated MSE')
plt.show()

# A11 + 12:
clf = Ridge(alpha=l_opt, fit_intercept=True)
clf.fit(X_train, y_train)
features = []
feats = np.argsort(np.absolute(np.array(clf.coef_))).flatten()
for co in feats[::-1]:
  features.append(X_train.columns[co])

coefs = np.sort(np.absolute(np.array(clf.coef_))).flatten()

ax = plt.gca()
ax.plot(features, coefs[::-1], linewidth = 2)


plt.xlabel("feature")
# Set ticks labels for x-axis
ax.set_xticklabels(features, rotation='vertical', fontsize=12)
plt.ylabel("absolute value")
plt.title("Feature absolute value coefficients")
plt.axis("tight")
plt.show()

#part 5

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
import updated_data_loading.preliminary as pre



test_data, train_data, raw_blood = pre.updated_preprocess()

plot3d(df = train_data, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = 'spread_score w.r.t. suger_levels and PCR_05')

X = train_data[['PCR_05', 'sugar_levels']]
y = train_data['spread_score']


alphas = np.logspace(start = -3, stop = 8, base = 10, num = 50 )
# alphas = range(100,250)
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
plt.grid('both')
plt.show()


opt_ridge = Ridge(alpha = val_opt_alpha)
predictions = opt_ridge.fit(X, y).predict(X)

# plot3d(df = train_data, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = "Opt ridge regressor's predictions (blue) vs true labels (red)", predictions = predictions)
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
plt.grid('both')
plt.show()

opt_poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', StandardScaler()),('Ridge', Ridge(alpha=val_opt_alpha, fit_intercept=True))])
predictions = opt_poly_reg.fit(X, y).predict(X)

plot3d(df = train_data, colX= 'sugar_levels', colY = 'PCR_05', colZ = 'spread_score', title = "Opt poly ridge regressor's predictions (blue) vs true labels (red)", predictions = predictions)

#part 6 

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

alphas = np.logspace(start = -3, stop = 8 , base = 10, num = 50 )
# alphas = np.arange(3,7,0.1)
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
plt.grid('both')
plt.show()

opt_poly_reg = Pipeline([('feature_mapping', PolynomialFeatures()), ('normalization', MinMaxScaler()),('Ridge', Ridge(alpha=val_opt_alpha, fit_intercept=True))])
predictions = opt_poly_reg.fit(X, y).predict(X)



#part 7 

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
linear_reggressor = LinearRegressor(0.1)
linear_reggressor.fit(X_train, y_train)
y_pred2 = linear_reggressor.predict(X_test)
print("linear:", mean_squared_error(y_test, y_pred2))
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


