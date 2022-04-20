import train as tr
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

model = tr.train_model()

# ct = model[0]
# print(ct)

while True:

    age = int(input("How old are you? \n"))
    sex = str(input("Which is your sex? If you are a male answer 'm', if you are a female answer with an 'f': \n"))

    if sex == 'm':
        sex = 'male'
    else:
        sex = 'female'

    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? If you smoke answer 'y', if you don't smoke answer with an 'n': \n"))

    if smoke == 'y':
        smoke = 'yes'
    else:
        smoke = 'no'

    bmi = float(input("What's is your bmi? \n"))
    region = str(input('Which is your residential area in the US? Options: northeast, southeast, southwest, northwest \n'))

    # Preprocess

    gbr = model[1] # the [1] refers to Gradient Boosting Regressor in the train file
    scaler = StandardScaler()

    ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] )] )
    x = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke," region":region}, index=[0])
    x_trans = ct.transform(x)
    x_scaled = scaler.transform(x_trans)

    # Predict

    prediction = (clf.predict(x_scaled) for clf in model[:-2])


   
    

    print(f"Prediction for your conditions: {prediction.mean()}")

