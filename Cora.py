import torch
import torchvision
import networkx as nx
import numpy as np
import pandas as pd
import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import data, dataset, in_memory_dataset

cora = Planetoid(".", "Cora", "public")
coraData = cora[0]
vectors = coraData.x
labels = coraData.y

train_x = coraData.x[:640].numpy()
train_y = coraData.y[:640].numpy()
train_edges = coraData.edge_index[:640].numpy()

test_x = coraData.x[-1000:].numpy()
test_y = coraData.y[-1000:].numpy()
test_edges = coraData.edge_index[-1000:].numpy()

#K-nn
# for k in [1, 3, 5, 7, 9, 11]:
knn = KNeighborsClassifier(11)
knn.fit(train_x, train_y)

knn_pred = knn.predict(test_x)
print(classification_report(test_y, knn_pred))

#MLP
layers = 50
activation = "relu"
solver = "sgd"
alpha = 0.0001
learning_rate = 0.1
max_iter = 200

mlp = MLPClassifier(
    hidden_layer_sizes=layers,
    activation=activation,
    solver=solver,
    alpha=alpha,
    learning_rate_init=learning_rate,
    max_iter=max_iter)

mlp.fit(train_x, train_y)
mlp_pred = mlp.predict(test_x)
print(classification_report(test_y, mlp_pred))


# SVM
svm = SVC(kernel="linear", gamma="auto")

svm.fit(train_x, train_y)
svm_pred = svm.predict(test_x)
print(classification_report(test_y, svm_pred))

edges = [[0 for col in range(2708)] for row in range(2708)]

# extract edges
for i in range(len(coraData.edge_index[0])):
    edges[coraData.edge_index[0][i]][coraData.edge_index[1][i]] = 1

new_x = []
for i in range(len(coraData.x)):
    new_x.append(np.append(coraData.x[i], edges[i]))

new_train_x = new_x[:640]
new_test_x = new_x[-1000:]

#K-nn
# for k in [1, 3, 5, 7, 9, 11]:
knn = KNeighborsClassifier(11)
knn.fit(new_train_x, train_y)

knn_pred = knn.predict(new_test_x)
print(classification_report(test_y, knn_pred))

#MLP
layers = 50
activation = "relu"
solver = "sgd"
alpha = 0.0001
learning_rate = 0.1
max_iter = 200

mlp = MLPClassifier(
    hidden_layer_sizes=layers,
    activation=activation,
    solver=solver,
    alpha=alpha,
    learning_rate_init=learning_rate,
    max_iter=max_iter)

mlp.fit(new_train_x, train_y)
mlp_pred = mlp.predict(new_test_x)
print(classification_report(test_y, mlp_pred))


# SVM
svm = SVC(kernel="linear", gamma="auto")

svm.fit(new_train_x, train_y)
svm_pred = svm.predict(new_test_x)
print(classification_report(test_y, svm_pred))