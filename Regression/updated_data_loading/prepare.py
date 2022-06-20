import pandas as pd
import numpy as np
import math
from numpy import nan
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(data, training_data):

    raw_data = data.copy()
    raw_training_data = training_data.copy()

    raw_data.reset_index(inplace = True)
    raw_training_data.reset_index(inplace = True)

    raw_data = select_features(raw_data)
    raw_training_data = select_features(raw_training_data)

    prepared_data = transform_features(raw_data)
    prepared_training_data = transform_features(raw_training_data)
   
    prepared_data, prepared_training_data = mean_imputate_features(prepared_data, prepared_training_data)
    prepared_data, prepared_training_data = median_imputate_features(prepared_data, prepared_training_data)
    # prepared_data, prepared_training_data = most_freq_imputate_features(prepared_data, prepared_training_data)

    prepared_training_data = select_features_after(prepared_training_data)
    prepared_data = select_features_after(prepared_data)

    prepared_data = normalize_features(prepared_data, prepared_training_data)

    prepared_data = prepared_data[sorted(prepared_data.columns)]

    #i added this for the part 1 of major 2 -N
    prepared_training_data = normalize_features(prepared_training_data,prepared_training_data)
    prepared_training_data = prepared_training_data[sorted(prepared_training_data.columns)]

    return prepared_data, prepared_training_data

def select_features(data):
  _data = data[['PCR_01', 'PCR_02', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_09', 'sport_activity', 'sugar_levels', 'symptoms','blood_type','sex','covid_score','spread_score']]
  return _data

def select_features_after(data):
  return data.drop(columns = ['low_appetite','cough'])

def transform_features(data):
  string_to_numeric(data)
  data = one_hot_encoding(data)
  unique_symptoms = get_symptoms(data)
  data = add_symptoms_features(data, unique_symptoms)
  string_to_numeric(data)

  return data

def string_to_numeric(data):
  data.replace({"High": 1, "Low": -1}, inplace=True)
  data.replace({"F": 1, "M": -1}, inplace=True)
  data.replace({True: 1, False: -1}, inplace=True)
  



def one_hot_encoding(data):
  data.replace({"A+": "A", "A-": "A", "B+": "Others",
                    "B-": "Others", "AB+": "Others", "AB-": "Others", "O+": "O", "O-": "O"}, inplace=True)
  blood_type_num = pd.get_dummies(data['blood_type'])
  joined_data = data.join(blood_type_num)
  joined_data.drop(['blood_type'], axis = 1, inplace = True)

  return joined_data


def get_symptoms(data):
  symptoms_list = data['symptoms'].unique()
  unique_symptoms = []
  for i in symptoms_list:
    if (type(i) is str):
      unique_symptoms.append(i.split(";")) 

  unique_symptoms = list(itertools.chain.from_iterable(unique_symptoms))
  unique_symptoms = pd.Series(unique_symptoms)
  unique_symptoms = unique_symptoms.unique()

  return unique_symptoms


def add_symptoms_features(data, unique_symptoms):
  
  symptoms_df = pd.DataFrame(index=range(data.shape[0]))
  for symptom in unique_symptoms:
    symptoms_df[symptom] = np.nan
  symptoms_df.fillna(0, inplace=True)
  symptoms_df[np.isnan(symptoms_df)] = 0

  
  joined_data = data.join(symptoms_df)

  for index in joined_data.index:
    if(type(joined_data['symptoms'][index]) is not str):
      continue
    for symptom in unique_symptoms:
      if (symptom in joined_data['symptoms'][index]):
        joined_data[symptom][index] = 1

  joined_data.drop(['symptoms'], axis = 1, inplace = True)
  
  return joined_data     

def mean_imputate_features(data, training_data):
  
    mean_features = ['PCR_01','PCR_02','PCR_04','PCR_05', 'PCR_06', 'PCR_09', 'sugar_levels','sport_activity','shortness_of_breath','sore_throat','fever']
    imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    training_data[mean_features] = imputer1.fit_transform(training_data[mean_features])
    data[mean_features] = imputer1.transform(data[mean_features])

    return data, training_data


def median_imputate_features(data, training_data):
  
    features = ['sport_activity']
    imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
    training_data[features] = imputer1.fit_transform(training_data[features])
    data[features] = imputer1.transform(data[features])

    return data, training_data

def most_freq_imputate_features(data, training_data):

    imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    training_data[['A', 'O', 'Others']] = imputer2.fit_transform(training_data[['A', 'O', 'Others']])
    data[['A', 'O', 'Others']] = imputer2.transform(data[['A', 'O', 'Others']])
    return data, training_data

def normalize_features(data, training_data):
  
  from sklearn import preprocessing

  scaler = preprocessing.MinMaxScaler().fit(training_data)

  scaled_data = scaler.transform(data)

  data.loc[:,:] = scaled_data

  return data
