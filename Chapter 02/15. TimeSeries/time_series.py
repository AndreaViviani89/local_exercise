# import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# Load the data

df = pd.read_csv('climate.csv')
df = df.drop(columns= 'Date Time')
# print(df.shape)
# print(df.head())
# print(df.info())

def get_sequence(data, seq_len, target_name):

    seq_list = []
    target_list = []

    for i in range(0, data.shape[0] - (seq_len + 1), seq_len + 1):

        seq = data[i : seq_len + i]
        target = data[target_name][seq_len + i]

        seq_list.append(seq)
        target_list.append(target)

    return np.array(seq_list), np.array(target_list)

x, y = get_sequence(df, seq_len= 6, target_name= 'T (degC)')

# print(x.shape)
# print(y.shape)


# Extract the feature and apply different statistics from different columns and more statistics in the same column

def get_feature(x):

    feature = []

    for i in range(x.shape[0]):

        mean_column_1 = np.mean(x [i, :, 0])                            # calculate the mean for the 1st column
        std_column_1 = np.std(x [i, : , 0])                             # calculate the std for the 1st column
        median_column_1 = np.median(x [i, :, 0])                        # calculate the median for the 1st column
        min_column_2 = np.min(x [i, :, 1])                              # calculate the min for the column n. 2
        mean_column_2 = np.mean(x [i, :, 1])                            # calculate the mean for the column n. 2
        max_column_4 = np.max(x [i, :, 3])                              # calculate the max for the column n. 4
        std_column_6 = np.std(x [i, :, 5])                              # calculate the std for the column n. 6
        max_column_6 = np.max(x [i, :, 5])                              # calculate the max for the column n. 6
        min_column_8 = np.min(x [i, :, 7])                              # calculate the min for the column n. 8
        mean_column_10 = np.mean(x [i, :, 9])                           # calculate the mean for the column n. 10

        # append all the feature in an empty list
        feature.append((mean_column_1, std_column_1, median_column_1, min_column_2, mean_column_2, max_column_4, std_column_6, max_column_6, min_column_8, mean_column_10))

    return np.array(feature)

x = get_feature(x)

# print(all_the_feature.shape)


# create the models dictionary
model_dict = {
                'SVR': SVR(),
                'Adaboost': AdaBoostRegressor(),
                'GMB': GradientBoostingRegressor(),
                'Catboost': CatBoostRegressor(),
}

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Create a pipeline
pip = {name: Pipeline([('scalar', StandardScaler()), ('regressor', model)]) 
                       for name, model in model_dict.items()}


results = pd.DataFrame({
                            'Model': [],
                            'MSE': [],
                            'MAB': [],
                            'R2 Score': [],
                            'Time': []
})


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



# plt.figure()
# plt.plot(np.linspace(1, y_test.shape[0],y_test.shape[0]), y_test, label='real', linewidth=1 )
# plt.plot(np.linspace(1, y_test.shape[0],y_test.shape[0]), pred, linestyle='dashed', label='prediction',linewidth=0.5 )
# plt.legend()
# plt.show()




'''      Model       MSE       MAB  R2 Score         Time
0  Catboost  0.307944  0.391440  0.994939    12.868042
1       GMB  0.313973  0.407220  0.994840    32.705123
2  Adaboost  0.718464  0.656608  0.988192    12.630261
3       SVR  1.151909  0.829247  0.981069  1140.583319'''