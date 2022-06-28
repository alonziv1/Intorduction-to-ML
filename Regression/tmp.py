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

print(test_data.head())

