import pandas as pd
import numpy as np



'''
pa = pd.Series([4, 6, -5, 3], index=["x1", "x2", "x3", "x4"])
print(pa)

pa_new = pd.Series([4, 6, -5, None], index=["x1", "x2", "x3", "x4"])
missing_val = pd.isnull(pa_new)

clean_data = pa_new[missing_val]

print(clean_data)

df = pd.DataFrame({"Data1": [10, 20, 30], "Data2": [40,50,60]})
norm_df = (df-df.min())/(df.max()-df.min())
print(norm_df)
'''

#ReIndex
data ={
"name": ["Maria", "Carla", "Juan", "Ana", "Sergio"],
"age": [15, 33, 12, 21, 45],
"gender": [True, True, False, True, False]
}

df = pd.DataFrame(data)
df.index = ["m", "c", "j", "a", "s"]
print(df) 

a = pd.DataFrame(data, index = ["m", "c", "j", "a", "s"])
print(a.loc["m" : "c"]) # conteggia solo le due lettere selezionate --> LOC include l'ultimo elemento

print(a.iloc[0:1]) # ILOC non conteggia l'ultimo elemento segnato, come nelle liste

print(a.drop(["m", "c"])) # DROP passi/escludi elementi

df = pd.DataFrame(data, index=['m','c','j','a','s'])
print(df[df.gender == False])

print(df.describe())

print(df['gender'].value_counts())







