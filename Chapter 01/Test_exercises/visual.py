import pandas as pd
import streamlit as st
import numpy as np

st.title("This is my first app!")
st.write("This is a table")

data = pd.DataFrame(np.random(10, 20), 
    columns = ('col %d'  % i 
        for i in range(20)))

st.write(data)



# Line chart
st.write("line chart")
st.line_chart(data)


#area chart
st.write("area chart")
st.area_chart(data)

#Histogram

st.write("histogram")
st.bar_chart(data)

#Map visualization
st.write("Map vis")
df = pd.DataFrame(
    np.random.randn(1000, 2)/[60,60] + [36.66, -121.6],
    columns = ["latitude", "longitude"])

st.map(df)