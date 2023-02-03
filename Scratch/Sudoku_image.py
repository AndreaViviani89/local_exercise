'''In questo script, la funzione extract_grid utilizza OpenCV per estrarre i contorni dei numeri da un'immagine
e quindi calcolare i numeri presenti in ogni cella.'''

import cv2
import numpy as np

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
            roi = thresh[y:y + h, x:x + w]
            _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
            grid[i][j] = int(np.sum(roi) / 255)

    return grid

def solve(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num
                        if solve(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

def is_valid(grid, row, col, num):
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    row_start = (row//3)*3
    col_start = (col//3)*3
    for i in range(3):
        for j in range(3):
            if grid[row_start + i][col_start + j] == num:
                return False
    return True

image = cv2.imread("sudoku.jpg")
grid = extract_grid(image)

if solve(grid):
    for row in grid:
        print(row)
else:
    print("No solution")
