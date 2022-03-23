import streamlit as st
import numpy as np
from sklearn import datasets



st.title("My Web App")

st.title("Comparing the best Ml classifier")

data_select = st.sidebar.selectbox("select your dataset", ("Wine data", "Iris data", "brest cancer data"))
st.write(data_select)

classifier_select = st.sidebar.selectbox("select your classifier", ("SVM", "Random forest", "KNN"))


def load_data(data_select):
    if data_select == "Wine data":
        df = datasets.load_wine()
    elif data_select == "Iris data":
        df = datasets.load_iris()
    else:
        df = datasets.load_brest_cancer()
    x = df.data
    y = df.target
    return x, y

x, y = load_data(data_select)
st.write("Shape of the data: ", x.shape)
st.write("Length of y: ", len(np.unique(y)))

def parameters(classifier_select):
    params = dict()
    if classifier_select == "SVM":
        c = st.sidebar.slider("c", 0.01, 10.0)
        params["c"] = c
    elif classifier_select == "KNN":
        k = st.sidebar.slider("k", 1, 15)
        params["k"] = k
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    return params

parameters(classifier_select)




