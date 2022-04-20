from turtle import shape
from train import Algorithm
from train import Main
import data_handler as dh 

x_train, x_test, y_train, y_test, ct = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

alg_model = Algorithm(x_train, x_test, y_train, y_test)
print(alg_model.check_the_shape())

rf = alg_model.rand_for()
print(rf[1])

# print(rf.accuracy)