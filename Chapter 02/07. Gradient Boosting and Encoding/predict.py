import train as tr
import pandas as pd
import numpy as np
model = tr.train_model()

ct = model[0]
print(ct)

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
    region = str(input('Which is your residential area in the US? \n'))

    # Preprocess

    # x_train = ct.fit_transform(x_train)
    # x_test = ct.transform(x_test)
    gbr = model[1]

    x = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke," region":region}, index=[0])

    x_trans = gbr.transform(x)



    # Predict

   
    

    print(f"Prediction for your conditions: {predictions.mean()}")