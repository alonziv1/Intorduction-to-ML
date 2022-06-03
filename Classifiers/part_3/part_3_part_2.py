from SoftSVM import SoftSVM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import visualize_clf
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


train_data= pd.read_csv("part_3_train_data_new.csv")
test_data= pd.read_csv("part_3_test_data_new.csv")

"""plot = sns.jointplot(data = train_data, x='PCR_05', y='sugar_levels', hue = 'spread')
plot.fig.suptitle("Marginal and joint distribution of PCR_05 and sugar_levels according to 'spread' label")
"""
X_train = train_data[['PCR_05','sugar_levels']]
y_train = train_data['spread']
X_test = test_data[['PCR_05','sugar_levels']]
y_test = test_data['spread']



svm_clf = Pipeline([('feature_mapping', PolynomialFeatures(3)), ('SVM', SoftSVM(C=1e3, lr=1e-4, batch_size=32))])
accuracies = []
for iteration in range(5):
    svm_clf.fit(X_train, y_train, SVM__max_iter=10000)
    accuracies.append(svm_clf.score(X_train, y_train))
    visualize_clf.visualize_clf(svm_clf, X_train.to_numpy(), y_train.to_numpy(), "boundries", "PCR_05" , "sugar_levels" )

print( accuracies)
visualize_clf.visualize_clf(svm_clf, X_train.to_numpy(), y_train.to_numpy(), "boundries", "PCR_05" , "sugar_levels" )