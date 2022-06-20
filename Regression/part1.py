import updated_data_loading.preliminary as pre
import seaborn as sns

# updated preprocess phase:
test_data, train_data, raw_blood = pre.updated_preprocess()

# A1:
temp_train = train_data.join(raw_blood)
temp_train.replace({"A+": "A", "A-": "A", "B+": "Others",
                    "B-": "Others", "AB+": "Others", "AB-": "Others", "O+": "O", "O-": "O"}, inplace=True)
rel = sns.kdeplot(data=temp_train, x='covid_score',hue='blood_type', common_norm=False).set(title='Covid Score by Blood Type')

