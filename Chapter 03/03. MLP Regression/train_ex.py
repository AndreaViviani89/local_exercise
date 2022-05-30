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

data = '03. MLP Regression/data/turkish_stocks.csv'

x_train, x_test, y_train, y_test = dh.load_data(pth=data)
x_train_batches, x_test_batches, y_train_batches, y_test_batches = dh.to_batches(x_train, x_test, y_train, y_test, batch_size=20)

optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
criterion = torch.nn.MSELoss()

def torch_fit(model, x_train, x_test, y_train, y_test, x_train_batches, x_test_batches, y_train_batches, y_test_batches, num_epochs):
    train_loss_list = []
    test_loss_list = []
    benchmark = 0.7
    for epoch in range(num_epochs):
        print(f"Current epoch: {epoch+1}/{num_epochs}")
        current_loss = 0
        for batch, (x_train_batches, y_train_batches) in enumerate(zip(x_train, y_train)):
            optimizer.zero_grad()
            train_pred = model(x_train_batches)

            train_loss = criterion(train_pred, y_train_batches)
            current_loss += train_loss.item()

            train_loss.backward()

            optimizer.step()
        train_loss_list.append(current_loss/x_train.shape[0])

        # test

        model.eval()
        with torch.no_grad():
            current_loss = 0
            for batch_t, (x_test_batches, y_test_batches) in enumerate(zip(x_test, y_test)):
                test_pred = model(x_test_batches)

                test_loss = criterion(test_pred, y_test_batches)
                current_loss += test_loss.item()

            test_loss_list.append(current_loss/x_test.shape[0])

            # check best model
            if test_loss_list[-1] > benchmark:
                torch.save(model, "model.pth")

        model.train()
    x_axis = list(range(num_epochs))
    plt.subplot(1,2,1)
    plt.plot(x_axis, train_loss_list, marker='o', label='Train loss')
    plt.plot(x_axis, test_loss_list, marker='o', label='Test loss')
    plt.legend(loc='best')
    plt.show()


model = torch_fit(model, x_train, x_test, y_train, y_test, x_train_batches, x_test_batches, y_train_batches, y_test_batches, num_epochs=30)
