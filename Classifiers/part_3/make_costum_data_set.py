from SoftSVM import SoftSVM
import matplotlib.pyplot as plt
import seaborn as sns
import verify_gradients
import numpy as np
import visualize_clf
import pandas as pd
from prepare import prepare_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#@title

def transform_labels(dataset):

    dataset['risk'].replace({1: 1, 0: -1}, inplace=True)
    dataset['spread'].replace({1: -1, 0: 1}, inplace=True)
    dataset['covid'].replace({1: 1, 0: -1}, inplace=True)

dataset = pd.read_csv("virus_data.csv")
train_data, test_data= train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=156)
test_data, train_data = prepare_data(test_data, train_data)
transform_labels(train_data)
transform_labels(test_data)



train_data.to_csv(r'C:\Users\alonz\Documents\GitHub\Intorduction-to-ML\Classifiers\part_4\part_4_train_data.csv', index = False)
test_data.to_csv(r'C:\Users\alonz\Documents\GitHub\Intorduction-to-ML\Classifiers\part_4\part_4_test_data.csv', index = False)

