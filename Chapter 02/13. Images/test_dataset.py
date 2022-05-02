import numpy as np
from PIL import Image
import pandas as pd
import math


df_test = pd.read_csv('fashion-mnist_test.csv')

print(df_test.head())
print(df_test.shape)

img_label = df_test.values[0][0]
img_ex = df_test.values[0][1:]

img_ex = img_ex.reshape((28,28))
print(img_ex.shape)

img = Image.fromarray(img_ex.astype(int))
img.show()