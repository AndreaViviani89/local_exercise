import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, num_hidden3, num_hidden4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.fc4 = nn.Linear(num_hidden3, num_hidden4)
        self.fc5 = nn.Linear(num_hidden4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward (self, x):
        layer1 = self.fc1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.fc2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.fc3(act2)
        act3 = self.sigmoid(layer3)
        layer4 = self.fc4(act3)
        act4 = self.sigmoid(layer4)
        layer5 = self.fc5(act4)
        out = self.sigmoid(layer5)
        return out


data = pd.read_csv('data.csv', header= None)
X = torch.tensor(data.drop(2, axis=1).values, dtype= torch.float)
y = torch.tensor(data[2].values, dtype= torch.float).view(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

criterion = nn.BCELoss()
neuralnet = NeuralNetwork(X.shape[1], 10, 10, 10, 10)
epochs = 1000

def fit(x_train, x_test, y_train, y_test, model, criterion, lr, num_epo):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    test_losses = []
    accuracy = []

    for epoch in range(num_epo):
        optimizer.zero_grad()
        pred = model(x_train)
        loss_value = criterion(pred, y_train)
        train_losses.append(loss_value.detach().item())
        # print(f'Epoch {epoch}, loss {loss_value.item():.2f}')
        loss_value.backward()
        optimizer.step()

        # Evaluate
        model.eval()

        with torch.no_grad():
            test_preds = model.forward(x_test)
            test_loss = criterion(test_preds, y_test)
            test_losses.append(test_loss.item())

            classes = test_preds > 0.5

            acc = sum(classes == y_test) / classes.shape[0]

        model.train()

        accuracy.append(acc)

        print(f'Epoch: {epoch + 1} | loss: {loss_value.item()} | test loss: {test_loss.item()} | accuracy: {acc}')

