#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn
import time

import matplotlib.pyplot as plt
import numpy as np

from model import Network #import your model here



# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train



model = Network(8)


pth = '03. MLP Regression/data/turkish_stocks.csv'

x_train, x_test, y_train, y_test = dh.load_data(pth)

optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
criterion = torch.nn.MSELoss()

epochs = 10

train = []
test = []

for epoch in range(epochs):
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = dh.to_batches(x_train, x_test, y_train, y_test, batch_size= 7)
    running_loss = 0
    running_loss_test = 0

    for n in range(x_train_batch.shape[0]):
        optimizer.zero_grad()
        pred = model.forward(x_train_batch[n])
        loss = criterion(pred, y_train_batch[n])

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_pred = model.forward(x_test_batch)
            test_loss = criterion(test_pred, y_test_batch)

