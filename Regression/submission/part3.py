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