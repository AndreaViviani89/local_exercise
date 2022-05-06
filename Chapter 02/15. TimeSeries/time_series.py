# import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



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
        target = data[target_name][seq_len + 1]

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
        std_column_1 = np.std(x [i, : , 0])
        median_column_1 = np.median(x [i, :, 0])
        min_column_2 = np.min(x [i, :, 1])
        mean_column_2 = np.mean(x [i, :, 1])
        max_column_4 = np.max(x [i, :, 3])
        std_column_6 = np.std(x [i, :, 5])
        max_column_6 = np.max(x [i, :, 5])
        min_column_8 = np.min(x [i, :, 7])
        mean_column_10 = np.mean(x [i, :, 9])

        # append all the feature in an empty list
        feature.append((mean_column_1, std_column_1, median_column_1, min_column_2, mean_column_2, max_column_4, std_column_6, max_column_6, min_column_8, mean_column_10))

    return np.array(feature)

all_the_feature = get_feature(x)

print(all_the_feature.shape)

        