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