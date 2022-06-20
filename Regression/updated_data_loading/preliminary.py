import pandas as pd
from sklearn.model_selection import train_test_split
# import prepare as p

def updated_preprocess():
    dataset = pd.read_csv('HW3_data.csv')
    raw_blood = dataset['blood_type']
    train_data, test_data= train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=156)
    test_data, train_data = p.prepare_data(dataset, train_data)
    return test_data, train_data, raw_blood

