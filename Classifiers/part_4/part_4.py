from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pandas as pd
import sys 
sys.path.insert(0, 'C:/Users/alonz/Documents/GitHub/Intorduction-to-ML/Classifiers/part_3')
import functions as f 
from visualize_clf import visualize_clf 
import numpy as np
import matplotlib as plt


fixed_C = 1e4

##data

train_data= pd.read_csv("part_4_train_data.csv")
test_data= pd.read_csv("part_4_test_data.csv") 

# f.plot_data(train_data,x_label = 'PCR_08',y_label = 'PCR_10', hue = 'risk', title ="title")

X_train = train_data[['PCR_08','PCR_10']]
y_train = train_data['risk']
X_test = test_data[['PCR_08','PCR_10']]
y_test = test_data['risk']


gamma_range  = np.logspace(-5, 4, 10)

for gamma in gamma_range:
    rbf_svc = SVC(kernel='rbf', C = fixed_C , gamma = gamma)
    rbf_svc.fit(X_train, y_train)
    visualize_clf(rbf_svc , X_train.to_numpy(), y_train.to_numpy(), "devision boundaries with gamma = " +str(gamma), xlabel = 'PCR_08',ylabel = 'PCR_10')