from cgi import test
from unittest import result
import pandas as pd
import numpy as np

data = pd.read_csv('climate.csv')
data = data.drop(['Date Time'], axis= 1)

print(data)
print(data.shape)

def pairing(data, seq_len = 6):
    
    x = []
    y = []

    for i in range(0, (data.shape[0] - seq_len+1), seq_len+1):

        seq = np.zeros((seq_len, data.shape[1]))
        
        for j in range(seq_len):

            seq[j] = data.values[i+j]

        x.append(seq.flatten())         # with .flatten we set x in one dimension
        y.append(data['T (degC)'][i + seq_len])


    return np.array(x), np.array(y)



print(data.shape)

x, y = pairing(data)

print(x.shape)

print(y[0:3])
# print(y[1])

# print(y)
# print(y[1])


# Import useful libraries

import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn import metrics


# create the models dictionary
model_dict = {
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'SVR': SVR(),
                'Adaboost': AdaBoostRegressor(),
                'GMB': GradientBoostingRegressor(),
                'Catboost': CatBoostRegressor(),
}

# print(model_dict)

time_split = TimeSeriesSplit(n_splits=5)    # the default value is 5

results = pd.DataFrame({
                            'Model': [],
                            'MSE': [],
                            'MAB': [],
                            'R2 Score': [],
                            'Time': []
})

for train_index, test_index in time_split.split(x):
    scaler = StandardScaler()

    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    print(x_train.shape)
    print(x_test.shape)    

    # Scale the data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    for model_name, model in model_dict.items():
        start_time = time.time()
        model.fit(x_train, y_train)
