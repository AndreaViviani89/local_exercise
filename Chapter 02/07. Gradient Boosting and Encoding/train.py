import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_validate
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


x_train, x_test, y_train, y_test, ct = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

# Create the main class Algorithm
# class Algorithm:
#     def __init__(self, x_train, x_test, y_train, y_test):
#         self.x_train = x_train
#         self.x_test = x_test
#         self.y_train = y_train
#         self.y_test = y_test

#     #Check the shape
#     def check_the_shape(self):
#         x_train = self.x_train.shape
#         x_test = self.x_test.shape
#         y_train = self.y_train.shape
#         x_test = self.y_test.shape
#         return x_train, x_test, y_train, y_test


# #Instantiate the XGB Regressor
# class XGB (Algorithm):
#     def __init__(self, x_train, x_test, y_train, y_test):
#         super().__init__(x_train, x_test, y_train, y_test)
#         xgb_reg = XGBRegressor()
#         xgb_regr = xgb_reg.fit(x_train,y_train)
#         return xgb_regr

# # Instantiate the Gradient Boosting Regressor
# class grad_boost (Algorithm):
#     def __init__(self, x_train, x_test, y_train, y_test):
#         super().__init__(x_train, x_test, y_train, y_test)
#         grad_boosting_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
#         grad_boost = grad_boosting_reg.fit(x_train, y_train)
#         return grad_boost

# # Instantiate the random forest model
# class rand_for (Algorithm):
#     def __init__(self, x_train, x_test, y_train, y_test):
#         super().__init__(x_train, x_test, y_train, y_test)
#         rf = RandomForestRegressor(n_estimators= 50, random_state=0)
#         rand_for = rf.fit(x_train, y_train)
#         accuracy=rand_for.score(x_train, y_train)
#         print(f'Accuracy for random forest: {accuracy}')
#         return rand_for

# # Instantiate the AdaBoost
# class adaboost (Algorithm):
#     def __init__(self, x_train, x_test, y_train, y_test):
#         super().__init__(x_train, x_test, y_train, y_test)
#         ada = AdaBoostRegressor(random_state=0, n_estimators=100) # by default is set on 50, I tried with 100 of estimators
#         adaboost = ada.fit(x_train, y_train)
#         accuracy=adaboost.score(x_train, y_train)
#         return adaboost


# test = rand_for(x_train, x_test, y_train, y_test)
# print(test.accuracy)




# try in another way

class Main:
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
        y_test = self.y_test.shape
        return x_train, x_test, y_train, y_test

    def cross_validation(self):
        self.cv=cross_validate(self.x_train, self.y_train, return_estimator=True, cv=5)
        # need to figure it out 

class Algorithm (Main):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
    
    #Instantiate the XGB Regressor
    def XGB(self):
        xgb_reg = XGBRegressor()
        xgb_regr = xgb_reg.fit(x_train,y_train)
        accuracy= xgb_regr.score(x_train, y_train)
        # print(f'Accuracy for XGB: {accuracy}')
        return xgb_regr, accuracy

    # Instantiate the Gradient Boosting Regressor
    def grad_boost(self):
        grad_boosting_reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
        grad_boost = grad_boosting_reg.fit(x_train, y_train)
        accuracy= grad_boost.score(x_train, y_train)
        # print(f'Accuracy for Gradient Boosting Regressor: {accuracy}')
        return grad_boost, accuracy
    
    # Instantiate the random forest model
    def rand_for(self):
        rf = RandomForestRegressor(n_estimators= 50, random_state=0)
        rand_for = rf.fit(x_train, y_train)
        accuracy=rand_for.score(x_train, y_train)
        # print(f'Accuracy for random forest: {accuracy}')
        return rand_for, accuracy
    
    # Instantiate the AdaBoost
    def adaboost(self):
        ada = AdaBoostRegressor(random_state=0, n_estimators=100) # by default is set on 50, I tried with 100 of estimators
        adaboost = ada.fit(x_train, y_train)
        accuracy=adaboost.score(x_train, y_train)
        # print(f'Accuracy for adaboost: {accuracy}')
        return adaboost, accuracy

    