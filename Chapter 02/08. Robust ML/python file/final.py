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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from data_handler import x, y, x_test, pipelines, tree_prepro


# dictionary with some models

def models():
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Extra Trees": ExtraTreesClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "Skl GBM": GradientBoostingClassifier(random_state=0),
    "Skl HistGBM": HistGradientBoostingClassifier(random_state=0),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0),
    "CatBoost": CatBoostClassifier(random_state=0)
    }

    tree_classifiers = {name: pipeline.make_pipeline(pipelines(), model) for name, model in tree_classifiers.items()}
    return tree_classifiers

