# A27:
from sklearn.metrics import mean_squared_error

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

