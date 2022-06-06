from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pandas as pd
import sys 
sys.path.insert(0, 'C:/Users/alonz/Documents/GitHub/Intorduction-to-ML/Classifiers/part_3')
import functions as f 
from visualize_clf import visualize_clf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import seaborn as sns


#prepeations
fixed_C = 1e4
train_data= pd.read_csv("part_4_train_data.csv")
test_data= pd.read_csv("part_4_test_data.csv") 
# f.plot_data(train_data,x_label = 'PCR_08',y_label = 'PCR_10', hue = 'risk', title ="title")
X_train = train_data[['PCR_08','PCR_10']]
y_train = train_data['risk']
X_test = test_data[['PCR_08','PCR_10']]
y_test = test_data['risk']

def find_optimal_gamma():

    gamma_range  = np.logspace(-5, 4, 10)
    for gamma in gamma_range:
        rbf_svc = SVC(kernel='rbf', C = fixed_C , gamma = gamma)
        rbf_svc.fit(X_train, y_train)
        visualize_clf(rbf_svc , X_train.to_numpy(), y_train.to_numpy(), "devision boundaries with gamma = " +str(gamma), xlabel = 'PCR_08',ylabel = 'PCR_10')


def find1OptimalParam():

    gamma_range  = np.arange(30,1000,step  = 10)
    max = gamma_range[-1]
    min = max
    max_result = 0
    all_scores = []

    for gamma in gamma_range:
        rbf_svc = SVC(kernel='rbf', C = gamma , gamma = gamma)
        scores = cross_validate(rbf_svc, X_train, y_train, cv=2)
        mean = scores['test_score'].mean()
        print(mean, "for: ", gamma)
        all_scores.append(mean)
        if(mean > 0.7 and min == max):
            min = gamma
        if(mean > 0.9):
            max_result = gamma
            break
    print("the requested range : (",min,",",max_result,")")
    plt.plot(gamma_range, all_scores ) 
    plt.title( "gamma parameter - searching optimal range")   
    plt.xlabel("gamma value")
    plt.ylabel("2-cv mean score")
    plt.show()


def find2OptimalParam():

    C_min = 50
    C_max = 150
    gamma_min = 5
    gamma_max = 10
    rbf_clf = SVC(kernel='rbf')
    clf = Pipeline(steps=[('clf', rbf_clf)])
    C_range = np.arange(C_min, C_max, step = 5)
    gamma_range = range(gamma_min, gamma_max)
    parameters = dict(clf__C = C_range, clf__gamma = gamma_range)
    clf_GS = GridSearchCV(clf, parameters, cv=8, return_train_score=True)
    clf_GS.fit(X_train, y_train)
    print('Best C:', clf_GS.best_estimator_.get_params()['clf__C'])
    print('Best gamma:', clf_GS.best_estimator_.get_params()['clf__gamma'])

    mean_scores = clf_GS.cv_results_['mean_test_score']
    mean_scores = np.array(mean_scores).reshape(len(C_range),len(gamma_range))
    fig, ax = plt.subplots(figsize=(len(C_range),len(gamma_range)))
    ax = sns.heatmap(data= mean_scores, cbar = True, xticklabels = list(gamma_range),yticklabels = list(C_range) )
    ax.set(xlabel='gamma values', ylabel='C values', title='Mean validation accuracy')

    fig2, ax2 = plt.subplots(figsize=(12,12))
    mean_train_scores = clf_GS.cv_results_['mean_train_score']
    mean_train_scores = np.array(mean_train_scores).reshape(len(C_range),len(gamma_range))
    ax2 = sns.heatmap(data= mean_train_scores, cbar = True, xticklabels = list(gamma_range),yticklabels = list(C_range))
    ax2.set(xlabel='gamma values', ylabel='C values', title='Mean training accuracy')

    plt.show()


find2OptimalParam()
rbf_clf = SVC(kernel='rbf', C = 70 , gamma = 8)
rbf_clf.fit(X_train, y_train)
visualize_clf(rbf_clf , X_train.to_numpy(), y_train.to_numpy(), "devision boundaries with optimal parameters, score is ", xlabel = 'PCR_08',ylabel = 'PCR_10')
print(rbf_clf.score(X_test,y_test ))




