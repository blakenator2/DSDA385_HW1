import pandas as pd

columns= ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
training = pd.read_csv(r'adult\adult.data', names= columns,
                        skipinitialspace=True, header=None)

training = training[training['workclass'] != '?']
training = training[training['occupation'] != '?']
training = training[training['native-country'] != '?']

columns= ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
testing = pd.read_csv(r'adult\adult.test', names= columns,
                    skipinitialspace=True, header=None)

testing = testing[testing['workclass'] != '?']
testing = testing[testing['occupation'] != '?']
testing = testing[testing['native-country'] != '?'] #Remove all missing values, will make training much easier

training = training.sample(frac=0.7957032027) #I want 24,000 samples
testing = testing.sample(frac=0.53117322886) #I want 8,000 samples
#75/25 split, good for ML

training.loc[training['income'] == "<=50K" , 'income'] = '0'
training.loc[training['income'] == ">50K" , 'income'] = '1'

testing.loc[testing['income'] == "<=50K." , 'income'] = '0'
testing.loc[testing['income'] == ">50K." , 'income'] = '1'

training.to_csv(r'adult\adultTrain.csv')
testing.to_csv(r'adult\adultTest.csv')

training = pd.read_csv(r'adult\adultTrain.csv')

X_train= training[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
y_train = training['income']

testing = pd.read_csv(r'adult\adultTest.csv')

X_test= testing[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
y_test = testing['income']

my_dict={
"Private":0,
"Self-emp-not-inc":1,
"Self-emp-inc":2,
"Federal-gov":3,
"Local-gov":4,
"State-gov":5,
"Without-pay":6,
"Never-worked":7,

"Bachelors":0,
"Some-college":1,
"11th":2,
"HS-grad":3,
"Prof-school":4,
"Assoc-acdm":5,
"Assoc-voc":6,
"9th":7,
"7th-8th":8,
"12th":9,
"Masters":10,
"1st-4th":11,
"10th":12,
"Doctorate":13,
"5th-6th":14,
"Preschool":15,

'Married-civ-spouse': 0,
'Divorced': 1,
'Never-married': 2,
'Separated': 3,
'Widowed': 4,
'Married-spouse-absent': 5,
'Married-AF-spouse': 6,

'Tech-support': 0,
'Craft-repair': 1,
'Other-service': 2,
'Sales': 3,
'Exec-managerial': 4,
'Prof-specialty': 5,
'Handlers-cleaners': 6,
'Machine-op-inspct': 7,
'Adm-clerical': 8,
'Farming-fishing': 9,
'Transport-moving': 10,
'Priv-house-serv': 11,
'Protective-serv': 12,
'Armed-Forces': 13,

'Wife': 0,
'Own-child': 1,
'Husband': 2,
'Not-in-family': 3,
'Other-relative': 4,
'Unmarried': 5,

'White': 0,
'Asian-Pac-Islander': 1,
'Amer-Indian-Eskimo': 2,
'Other': 3,
'Black': 4,

"Male":0,
"Female":1,

'United-States': 0,
'Cambodia': 1,
'England': 2,
'Puerto-Rico': 3,
'Canada': 4,
'Germany': 5,
'Outlying-US(Guam-USVI-etc)': 6,
'India': 7,
'Japan': 8,
'Greece': 9,
'South': 10,
'China': 11,
'Cuba': 12,
'Iran': 13,
'Honduras': 14,
'Philippines': 15,
'Italy': 16,
'Poland': 17,
'Jamaica': 18,
'Vietnam': 19,
'Mexico': 20,
'Portugal': 21,
'Ireland': 22,
'France': 23,
'Dominican-Republic': 24,
'Laos': 25,
'Ecuador': 26,
'Taiwan': 27,
'Haiti': 28,
'Columbia': 29,
'Hungary': 30,
'Guatemala': 31,
'Nicaragua': 32,
'Scotland': 33,
'Thailand': 34,
'Yugoslavia': 35,
'El-Salvador': 36,
'Trinadad&Tobago': 37,
'Peru': 38,
'Hong': 39,
'Holand-Netherlands': 40
}


X_train = X_train.replace(my_dict)
X_test = X_test.replace(my_dict)

X_train = pd.read_csv(r'adult\xtrain.csv')
y_train = pd.read_csv(r'adult\ytrain.csv')
X_test = pd.read_csv(r'adult\xtest.csv')
y_test = pd.read_csv(r'adult\ytest.csv')

X_test['age'] = X_test['age'].astype(int)
y_train = y_train['income']
y_test = y_test['income']

X_train.to_csv(r'adult\xtrain.csv')
y_train.to_csv(r'adult\ytrain.csv')
X_test.to_csv(r'adult\xtest.csv')
y_test.to_csv(r'adult\ytest.csv')