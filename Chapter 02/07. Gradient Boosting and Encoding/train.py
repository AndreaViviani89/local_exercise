import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor



# def train_model():

#     x_train, x_test, y_train, y_test, ct, scaler = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

#     # As a good practice check all the shape
#     # print(x_train.shape)
#     # print(x_test.shape)
#     # print(y_test.shape)
#     # print(y_train.shape)

#     #Instantiate the XGB Regressor
#     xgb_reg = XGBRegressor()
#     xgb_reg.fit(x_train,y_train)

#     # Instantiate the Gradient Boosting Regressor
#     grad_boosting_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
#     grad_boosting_reg.fit(x_train, y_train)

#     # Instantiate the random forest model
#     rf = RandomForestRegressor(n_estimators= 50, random_state=0)
#     rf.fit(x_train, y_train)

#     # Instantiate the AdaBoost
#     ada = AdaBoostRegressor(random_state=0, n_estimators=100) # by default is set on 50, I tried with 100 of estimators
#     ada.fit(x_train, y_train)


#     return xgb_reg, grad_boosting_reg, rf, ada, ct, scaler 

# train_model()


x_train, x_test, y_train, y_test = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

# Create the main class Algorithm
class Algorithm:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    #Check the shape
    def check_the_shape(self):
        x_train = self.x_train.shape
        x_test = self.x_test.shape
        y_train = self.y_train.shape
        x_test = self.y_test.shape
        return x_train, x_test, y_train, y_test


#Instantiate the XGB Regressor
class XGB (Algorithm):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        xgb_reg = XGBRegressor()
        xgb_regr = xgb_reg.fit(x_train,y_train)
        return xgb_regr

# Instantiate the Gradient Boosting Regressor
class grad_boost (Algorithm):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        grad_boosting_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
        grad_boost = grad_boosting_reg.fit(x_train, y_train)
        return grad_boost

# Instantiate the random forest model
class rand_for (Algorithm):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        rf = RandomForestRegressor(n_estimators= 50, random_state=0)
        rand_for = rf.fit(x_train, y_train)
        return rand_for

# Instantiate the AdaBoost
class adaboost (Algorithm):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        ada = AdaBoostRegressor(random_state=0, n_estimators=100) # by default is set on 50, I tried with 100 of estimators
        adaboost = ada.fit(x_train, y_train)
        return adaboost
