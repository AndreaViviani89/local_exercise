import numpy as np
from PIL import Image
import pandas as pd



img = Image.open('5 project_data set.JPG')

print(img)
print(img.size)


arr = np.array(img, dtype=float)
print(arr.shape)
# img.show()


img2 = img.convert('L')     # BLACK AND WHITE
arr2 = np.array(img2, dtype=float)
print(arr2.shape)
img2.show()


# Resize
img3 = img.resize((255,255))     # BLACK AND WHITE
arr3 = np.array(img2, dtype=float)
print(arr3.shape)
# img3.show()


