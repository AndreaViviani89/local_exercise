import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import time

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor

'''
Metadata

"timestamp" - timestamp field for grouping the data
"cnt" - the count of a new bike shares
"t1" - real temperature in C
"t2" - temperature in C "feels like"
"hum" - humidity in percentage
"windspeed" - wind speed in km/h
"weathercode" - category of the weather
"isholiday" - boolean field - 1 holiday / 0 non holiday
"isweekend" - boolean field - 1 if the day is weekend
"season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

"weathe_code" category description:
1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 
2 = scattered clouds / few clouds 
3 = Broken clouds 
4 = Cloudy 
7 = Rain/ light Rain shower/ Light rain 
10 = rain with thunderstorm 
26 = snowfall 
94 = Freezing Fog

'''


"""Load the data"""
data = pd.read_csv(r'data\london_merged.csv')
# print(data)

# Exploring the data
print(data.shape)
print(data.info())

np.random.seed(0)

# Check the structure of 'timestamp'
print(data['timestamp'][:5])


"""taking the hours, months and years out of timestamp and dropping timestamp column"""
data['year'] = data['timestamp'].apply(lambda row: row[:4])
# Check data['year']
print(data['year'][:5])

data['month'] = data['timestamp'].apply(lambda row: row.split('-')[2][:2])
# Check data['month']
print(data['month'][:5])

data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:])
# Check data['hour']
print(data['hour'][:5])

data.drop('timestamp', axis=1, inplace=True)
# Check the shape
print(data.shape)
# print(data[:5]) # check the new data


def feature_enhancement(data):
    
    df = data   # make a DataFrame copy

    for season in data['season'].unique():
        seasonal_data = df[df['season'] == season]
        hum_mean = seasonal_data['hum'].mean()
        wind_speed_mean = seasonal_data['wind_speed'].mean()
        t1_mean = seasonal_data['t1'].mean()
        t2_mean = seasonal_data['t2'].mean()

        for i in df[df['season'] == season].index:
            if np.random.randint(2) == 1:
                df['hum'].values[i] += hum_mean/10
            else:
                df['hum'].values[i] -= hum_mean/10

            if np.random.randint(2) == 1:
                df['wind_speed'].values[i] += wind_speed_mean/10
            else:
                df['wind_spped'].values[i] -= wind_speed_mean/10

            if np.random.randint(2) == 1:
                df['t1'].values[i] += t1_mean/10
            else:
                df['t1'].values[i] -= t1_mean/10

            if np.random.randint(2) == 1:
                df['t2'].values[i] += t2_mean/10
            else:
                df['t2'].values[i] -= t2_mean/10

    return df
