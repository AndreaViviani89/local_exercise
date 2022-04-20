from train import Algorithm
from train import rand_for
import data_handler as dh 

x_train, x_test, y_train, y_test = dh.get_data("C:/Users/andre/Documents/Strive_repository/local_exercise/Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

alg_model = Algorithm(x_train, x_test, y_train, y_test)
print(alg_model.check_the_shape())

