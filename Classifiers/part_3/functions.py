from imghdr import tests
from SoftSVM import SoftSVM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import visualize_clf
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.datasets import make_moons


def make_moons(plot = False):

    X_moons, y_moons = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=156)
    y_moons = ((2 * y_moons) - 1)[:, None] 
    
    if (plot):
        print(f"{X_moons.shape}, {y_moons.shape}") 
        plt.figure(), plt.grid(alpha=0.5), plt.title("Synthetic moon dataset") 
        _ = sns.scatterplot(x=X_moons[:, 0], y=X_moons[:, 1], hue=y_moons[:, 0])

    return X_moons, y_moons

def train_svm(X_moons, y_moons, lr = 2e-4):
    clf = SoftSVM(C=1, lr=2e-4) 
    losses, accuracies = clf.fit_with_logs(X_moons, y_moons, max_iter=3000) 
    plt.figure(figsize=(13, 6)) 
    plt.subplot(121), plt.grid(alpha=0.5), plt.title ("Loss") 
    plt.semilogy(losses), plt.xlabel("itration"), plt.ylabel("loss") 
    plt.subplot(122), plt.grid(alpha=0.5), plt.title ("Accuracy") 
    plt.plot(accuracies), plt.xlabel("itration"), plt.ylabel("accuracy") 
    plt.tight_layout()
    plt.show()

def plot_data(data,x_label, y_label, hue, title):
    plot = sns.jointplot(data = data, x=x_label, y=y_label, hue =hue)
    plot.fig.suptitle(title)
    plt.show()
 
def split(train_data, test_data):
    X_train = train_data[['PCR_05','sugar_levels']]
    y_train = train_data['spread']
    X_test = test_data[['PCR_05','sugar_levels']]
    y_test = test_data['spread']

    return X_train, y_train, X_test, y_test

def pre_tuning_svm_clf(X_train, y_train):
    svm_clf = Pipeline([('feature_mapping', PolynomialFeatures()), ('SVM', SoftSVM(C=1e4, lr=1e-3, batch_size=32))])
    scores = np.array([])
    for iteration in range(5):
        svm_clf.fit(X_train, y_train, SVM__max_iter=10000)
        scores = np.append(scores, svm_clf.score(X_train, y_train))
        visualize_clf.visualize_clf(svm_clf, X_train.to_numpy(), y_train.to_numpy(), "decision regions pre-tuning - iteration  " + str(iteration +1) , "PCR_05" , "sugar_levels" )

    print("the scores are ", scores , "the mean is", scores.mean() ," standard deviation is ", scores.std() )

def tuning_C_hyperparameter(X_train,y_train, start ,end):

    train_max = 0
    validation_max = 0
    train_C_opt = 0
    validation_C_opt = 0
    power_values = np.linspace(start,end)
    C_range = np.power(10,power_values).astype(float)
    C_train_accuracies= []
    C_validation_accuracies= []
    cv = KFold(n_split= 10, random_state= 156)
    for C in C_range:
        svm_clf = Pipeline([('feature_mapping', PolynomialFeatures()), ('SVM', SoftSVM(C=C, lr=1e-3, batch_size=32))])
        scores = cross_validate(svm_clf, X_train, y_train, cv=cv, return_train_score = True)
        train_mean = scores['train_score'].mean()
        validation_mean = scores['test_score'].mean()
        if(train_mean > train_max):
            train_max = train_mean
            train_C_opt = C
        if(validation_mean > validation_max):
            validation_max = validation_mean
            validation_C_opt = C
        C_train_accuracies.append(train_mean)
        C_validation_accuracies.append(validation_mean)
    plt.plot(power_values, C_train_accuracies, label="train accuracy")
    plt.plot(power_values, C_validation_accuracies , label="validation accuracy")
    plt.xlabel('Value of C (logarithmic scale, base =10)', fontsize="xx-large")
    plt.ylabel('cross validated Accuracy', fontsize="xx-large")
    plt.title('Using 10-folds cross-validation to tune the C hyperparameter for soft SVM classifier',fontsize="xx-large")
    plt.legend(fontsize="xx-large")
    plt.show()

    print("the best C w.r.t to validation accuracy is: ", validation_C_opt, "that achived a score of ", validation_max ,"\n")
    print("the best C w.r.t to train accuracy is: ", train_C_opt, "that achived a score of ", train_max,"\n")

    return validation_C_opt 

def train_and_test_tuned_C_svm_clf (X_train, y_train, X_test, y_test ,optimal_C  ):
    
    svm_clf = Pipeline([('feature_mapping', PolynomialFeatures()), ('SVM', SoftSVM(C=optimal_C, lr=1e-3, batch_size=32))])
    svm_clf.fit(X_train, y_train, SVM__max_iter=10000)

    visualize_clf.visualize_clf(svm_clf, X_test.to_numpy(), y_test.to_numpy(), "decision regions after-tuning ", "PCR_05" , "sugar_levels" )
    test_accuracy = svm_clf.score(X_test, y_test)
    print("the test accuracy is ", test_accuracy)

