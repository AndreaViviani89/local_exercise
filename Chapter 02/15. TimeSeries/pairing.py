from cgi import test
from tokenize import group
from unittest import result
import pandas as pd
import numpy as np

data = pd.read_csv('climate.csv')
data = data.drop(['Date Time'], axis= 1)

# print(data)
# print(data.shape)

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


x, y = pairing(data)

# print(x.shape)

# print(y[0:3])
# print(y[1])

# print(y)
# print(y[1])


# Get some feature: 
def features(data):
    new_df = []

    for i in range(data.shape[0]):          # we're getting each group

        groups = []     # empty list, we'll use it to append the results

        for k in range(data.shape[2]):          # we're getting each column and each group

            groups.append(np.mean(data[i][:, k]))
            groups.append(data[i][:, k][-1])
        
        new_df. append(groups)

    return np.array(new_df)



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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    # print(x_train.shape)
    # print(x_test.shape)    

    # Scale the data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    for model_name, model in model_dict.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        finish_time = time.time() - start_time

        pred = model.predict(x_test)

        results = results.append({
                                    'Model': model_name,
                                    'MSE': mean_squared_error(y_test, pred),
                                    'MAB': mean_absolute_error(y_test, pred),
                                    'R2 Score': r2_score(y_test, pred),
                                    'Time': finish_time},
                                    ignore_index=True)


results_sort = results.sort_values(by=['MSE'], ascending= True, ignore_index= True)
print(results_sort)



'''
            Model       MSE       MAB  R2 Score        Time
0   Random Forest  0.039449  0.129422  0.999393  433.847394
1   Random Forest  0.041701  0.133960  0.999287  844.501935
2             SVR  0.050544  0.141369  0.999136  368.891006
3             GMB  0.053628  0.163174  0.999174  161.879771
4   Random Forest  0.054916  0.151089  0.999221  274.558446
5   Random Forest  0.055179  0.148735  0.999122  632.246750
6             GMB  0.056155  0.166306  0.999040  269.022847
7        Catboost  0.056379  0.167579  0.999036   30.946549
8        Catboost  0.065136  0.176813  0.998997   25.794537
9   Random Forest  0.068426  0.164461  0.999126  122.880847
10            GMB  0.070456  0.180650  0.998879  217.901066
11            GMB  0.071540  0.182854  0.998985   97.406763
12            GMB  0.078619  0.191083  0.998996   52.656074
13       Catboost  0.081640  0.190452  0.998702   31.361436
14  Decision Tree  0.083888  0.193804  0.998708    6.840537
15  Decision Tree  0.087626  0.199180  0.998501   13.438639
16       Catboost  0.103006  0.210888  0.998538   21.790937
17  Decision Tree  0.104488  0.218234  0.998517    4.589422
18  Decision Tree  0.107898  0.217909  0.998284   10.514102
19  Decision Tree  0.133948  0.239234  0.998289    2.176627
20       Catboost  0.153779  0.251338  0.998036   18.099894
21            SVR  0.160437  0.186734  0.997530  122.925118
22            SVR  0.228329  0.204888  0.996369  249.169852
23       Adaboost  0.360824  0.447088  0.994261   97.660540
24       Adaboost  0.400137  0.481087  0.994890   18.264279
25       Adaboost  0.438385  0.512101  0.992503  120.159953
26       Adaboost  0.440872  0.526485  0.993211   69.462285
27            SVR  0.454998  0.268848  0.993542   54.611922
28       Adaboost  0.495972  0.567791  0.992960   41.989712
29            SVR  0.976010  0.357487  0.987536   14.708355
'''