from UCICNN import CNNmodel
from UCIMLP import MLPmodel
from CIFARCNN import CifarCNNmodel
from CifarPreprocessing import getData
import pandas as pd

X_train = pd.read_csv(r'adult\xtrain.csv')
y_train = pd.read_csv(r'adult\ytrain.csv')
X_test = pd.read_csv(r'adult\xtest.csv')
y_test = pd.read_csv(r'adult\ytest.csv')

y_test = y_test['income']
y_train = y_train['income']

MLPmodel(X_train, y_train, X_test, y_test, 64, 10)
CNNmodel(X_train, y_train, X_test, y_test, 64, 8)

# X_train, y_train, X_test, y_test = getData()
# CifarCNNmodel(X_train, y_train, X_test, y_test, 64, 12)