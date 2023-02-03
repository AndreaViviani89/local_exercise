'''In questo script, la funzione extract_grid utilizza un modello di deep learning 
(caricato dal file sudoku_model.h5) per riconoscere i numeri in ogni cella. '''


import cv2
import numpy as np
import tensorflow as tf

def extract_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 5 and h >= 20:
            cells.append((x, y, w, h))

    cells = sorted(cells, key=lambda x: x[0])
    cells = sorted(cells, key=lambda x: x[1])
    cells = np.array(cells)

    grid = np.zeros((9, 9), dtype="int")
    for i in range(9):
        for j in range(9):
            x, y, w, h = cells[i*9 + j]
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi / 255.0
            grid[i][j] = int(model.predict(roi[np.newaxis, :, :, np.newaxis]))

    return grid

model = tf.keras.models.load_model("sudoku_model.h5")

image = cv2.imread("sudoku.jpg")
grid = extract_grid(image)

if solve(grid):
    for row in grid:
        print(row)
else:
    print("No solution")
