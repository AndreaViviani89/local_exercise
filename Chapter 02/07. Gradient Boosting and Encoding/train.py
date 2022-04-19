import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor



def train_model():

    x_train, x_test, y_train, y_test = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

    xgb_r = XGBRegressor()
    gboosting_r = GradientBoostingRegressor(learning_rate=0.01, n_estimators=50)
    rf = RandomForestRegressor(n_estimators= 50, random_state=0)

    xgb_r.fit(x_train,y_train)
    gboosting_r.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    return xgb_r, gboosting_r, rf

train_model()

