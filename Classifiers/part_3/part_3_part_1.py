import matplotlib.pyplot as plt
from SoftSVM import SoftSVM
import seaborn as sns
import verify_gradients
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import visualize_clf


#Creating moon dataset
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=156)
y_moons = ((2 * y_moons) - 1)[:, None] 
print(f"{X_moons.shape}, {y_moons.shape}") 
"""plt.figure(), plt.grid(alpha=0.5), plt.title("Synthetic moon dataset") 
_ = sns.scatterplot(x=X_moons[:, 0], y=X_moons[:, 1], hue=y_moons[:, 0])

verify_gradients.compare_gradients(X_moons, y_moons, deltas=np.logspace(-9, -1, 12))

clf = SoftSVM(C=1, lr=2e-4) 
losses, accuracies = clf.fit_with_logs(X_moons, y_moons, max_iter=3000) 
plt.figure(figsize=(13, 6)) 
plt.subplot(121), plt.grid(alpha=0.5), plt.title ("Loss") 
plt.semilogy(losses), plt.xlabel("itration"), plt.ylabel("loss") 
plt.subplot(122), plt.grid(alpha=0.5), plt.title ("Accuracy") 
plt.plot(accuracies), plt.xlabel("itration"), plt.ylabel("accuracy") 
plt.tight_layout()
plt.show()
"""
svm_clf = Pipeline([('feature_mapping', PolynomialFeatures(3)), ('SVM', SoftSVM(C=1e3, lr=1e-4, batch_size=32))])

svm_clf.fit(X_moons, y_moons, SVM__max_iter=90000)
print(svm_clf.score(X_moons, y_moons))

visualize_clf.visualize_clf(svm_clf, X_moons, y_moons, "boundries", "PCR_05" , "sugar_levels" )