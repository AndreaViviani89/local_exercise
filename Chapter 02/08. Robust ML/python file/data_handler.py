import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn.pipeline import Pipeline      
from sklearn import pipeline        #Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder       # OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import compose
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config
from sklearn.compose import ColumnTransformer



set_config(display='diagram') # Useful for display the pipeline


# get the data and read the csv file

DATA_PATH = "C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/08. Robust ML/titanic/"

df      = pd.read_csv(DATA_PATH + "train.csv", index_col='PassengerId')
df_test = pd.read_csv(DATA_PATH + "test.csv",  index_col='PassengerId')

# check the shape
print("Train DataFrame:", df.shape)
print("Test DataFrame: ", df_test.shape)

# Make a lambda function --> the goal is extract the "title" from the column name
get_Title_from_Name = lambda x: x.split(',')[1].split('.')[0].strip()

df['Title'] = df['Name'].apply(get_Title_from_Name)
df_test['Title'] = df_test['Name'].apply(get_Title_from_Name)

# Title dictionary
title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


# Use map to apply the prevous dict
df["Title"] = df['Title'].map(title_dictionary)
df_test["Title"] = df_test['Title'].map(title_dictionary)

x_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin']) # # X_TEST DATA (NEW DATA)

cat_vars  = ['Sex', 'Embarked', 'Title']         # x.select_dtypes(include=[object]).columns.values.tolist()
num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age'] # x.select_dtypes(exclude=[object]).columns.values.tolist()


# create 2 pipelines

def pipelines():
    num_4_treeModels = pipeline.Pipeline([('SimplyImputer', SimpleImputer(missing_values=np.nan, strategy = 'mean'))])
    cat_4_treeModels =  pipeline.Pipeline([('SimplyImputer', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')),('Encoder', OrdinalEncoder())])

    tree_prepro = compose.ColumnTransformer(transformers =[ ('num', num_4_treeModels, num_vars),
                                                ('cat', cat_4_treeModels, cat_vars)],
                                                remainder='drop')

    return tree_prepro

