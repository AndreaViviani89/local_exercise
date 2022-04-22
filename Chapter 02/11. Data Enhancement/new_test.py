from turtle import distance
import pandas as pd
import numpy as np

# Create a new DataFrame

df = pd.DataFrame({'distance': [10, 25, 40, 100, 250], 'time':[1, 2, 4, 10, 25]})

# check df

print(df)

# add some feature

df['mean'] = df.apply(lambda x: x['distance'] / x['time'], axis= 1)

print(df)

# make a copy

df_new = df.copy()

df_new[['distance', 'time']] = df_new.apply(lambda x: x[['distance', 'time']] + np.random.random_sample()* 0.1, axis= 1) # np.random	Return random floats in the half-open interval [0.0, 1.0).

df = df.append(df_new.sample(int(df_new.shape[0]/4))) # 4 is equal to 25%
print(df)