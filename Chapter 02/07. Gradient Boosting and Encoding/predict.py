import train as tr
import pandas as pd
import numpy as np
model = tr.train_model()

while True:

    age = int(input("How old are you? \n"))
    sex = str(input("Which is your sex? If you are a male answer 'm', if you are a female answer with an 'f': \n"))

    if sex == 'm':
        sex = 'male'
    else:
        sex = 'female'

    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    bmi = float(input("What's is your bmi? \n"))
    region = str(input('Which is your residential area in the US? \n'))

    # Preprocess


    x = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke," region":region}, index=[0])



    # Predict

    predictions = np.array([clf.predict(x_scaled) for clf in model[:-2]])
    

    print(f"Prediction for your conditions: {predictions.mean()}")