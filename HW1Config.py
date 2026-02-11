from UCICNN import CNNmodel
from UCIMLP import MLPmodel
from CIFARCNN import CifarCNNmodel
from CIFARMLP import CifarMLPModel
from PCAMMLP import PcamMLPModel
from PCAMCNN import PcamCNNModel
from CifarPreprocessing import getData
from PCamPreprocessing import getPcamData
import pandas as pd

# X_train = pd.read_csv(r'adult\xtrain.csv')
# y_train = pd.read_csv(r'adult\ytrain.csv')
# X_test = pd.read_csv(r'adult\xtest.csv')
# y_test = pd.read_csv(r'adult\ytest.csv')

# y_test = y_test['income']
# y_train = y_train['income']

# MLPmodel(X_train, y_train, X_test, y_test, 64, 10, 0.001)
# CNNmodel(X_train, y_train, X_test, y_test, 64, 8, 0.001)

# X_train, y_train, X_test, y_test = getData()
# CifarCNNmodel(X_train, y_train, X_test, y_test, 64, 12, 0.001)
# CifarMLPModel(X_train, y_train, X_test, y_test, 64, 20, 0.001)

X_train, y_train, X_test, y_test = getPcamData()
#PcamMLPModel(X_train, y_train, X_test, y_test, 12, 0.001)
PcamCNNModel(X_train, y_train, X_test, y_test, 12, 0.001)