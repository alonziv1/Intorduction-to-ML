from cv2 import normalize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

ds = pd.read_csv('virus_data.csv')
ds = ds[['PCR_05', 'sugar_levels', 'spread']]
ds.replace({"High": 1, "Low": -1}, inplace=True)

train_ds, test_ds = train_test_split(ds, train_size = 0.8, test_size = 0.2, random_state= 156)

y_train = train_ds['spread']
y_test = test_ds['spread']

train_ds= train_ds[['PCR_05', 'sugar_levels']]
test_ds = test_ds[['PCR_05', 'sugar_levels']]

imputer1 = SimpleImputer()
train_ds[['PCR_05', 'sugar_levels']] = imputer1.fit_transform(train_ds[['PCR_05', 'sugar_levels']])
test_ds[['PCR_05', 'sugar_levels']] = imputer1.transform(test_ds[['PCR_05', 'sugar_levels']])

standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

scaled_train_ds = standard_scaler.fit_transform(train_ds)
train_ds.loc[:,:] = scaled_train_ds
scaled_test_ds= standard_scaler.transform(test_ds)
test_ds.loc[:,:] = scaled_test_ds

train_ds['spread']  = y_train
test_ds['spread']  = y_test

"""print ( "shape: ", train_ds.shape, '\n', train_ds.head(), '\n' )
print('pcr_values: ', train_ds[['PCR_05']].value_counts(), '\n')
print('suger_values: ', train_ds['sugar_levels'].value_counts(), '\n')
print('spread_values: ', train_ds['spread'].value_counts(), '\n')
"""



train_ds.to_csv(r'C:\Users\alonz\Documents\GitHub\Intorduction-to-ML\Classifiers\part_3\train_ds.csv', index = False)
test_ds.to_csv(r'C:\Users\alonz\Documents\GitHub\Intorduction-to-ML\Classifiers\part_3\test_ds.csv', index = False)

