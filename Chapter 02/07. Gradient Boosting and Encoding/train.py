import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor



def train_model():

    x_train, x_test, y_train, y_test, ct, scaler = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

    # As a good practice check all the shape
    print(x_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train.shape)

    #Instantiate the XGB Regressor
    xgb_reg = XGBRegressor()
    xgb_reg.fit(x_train,y_train)

    # Instantiate the Gradient Boosting Regressor
    grad_boosting_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
    grad_boosting_reg.fit(x_train, y_train)

    # Instantiate the random forest model
    rf = RandomForestRegressor(n_estimators= 50, random_state=0)
    rf.fit(x_train, y_train)

    # Instantiate the AdaBoost
    ada = AdaBoostRegressor(random_state=0, n_estimators=100) # by default is set on 50, I tried with 100 of estimators
    ada.fit(x_train, y_train)


    return xgb_reg, grad_boosting_reg, rf, ada, ct, scaler 

train_model()

