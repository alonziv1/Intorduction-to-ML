from cv2 import normalize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

ds = pd.read_csv('HW3_data.csv')
ds = ds[['PCR_05', 'sugar_levels', 'spread_score']]

train_ds, test_ds = train_test_split(ds, train_size = 0.8, test_size = 0.2, random_state= 156)

y = train_ds['spread_score'].copy()
train_ds.drop('spread_score', inplace = True)

imputer1 = SimpleImputer()
train_ds[['PCR_05', 'sugar_levels']] = imputer1.fit_transform(train_ds[['PCR_05', 'sugar_levels']])

standard_scaler = StandardScaler()
scaled_train_ds = standard_scaler.fit_transform(train_ds)
train_ds.loc[:,:] = scaled_train_ds

train_ds['spread_score']  = y 

train_ds.to_csv(r'C:\Users\alonz\Documents\GitHub\Intorduction-to-ML\Regression\train_ds.csv', index = False)

