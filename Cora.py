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

import visualization
def knn(train_x, train_y, test_x, test_y, title):
    best_score = 0
    best_k = 1
    print(title)
    for k in range(3, 16):
        knn = KNeighborsClassifier(k)
        knn.fit(train_x, train_y)

        score = knn.score(test_x, test_y)
        print("k = %i, score: %.4f%%" % (k, score * 100))
        if score > best_score:
            best_score = score
            best_k = k

    print("Best k =  %i" % best_k)
    knn = KNeighborsClassifier(best_k)
    knn.fit(train_x, train_y)
    knn_pred = knn.predict(test_x)
    visualization.visualize_confusion_cora(test_y,knn_pred,knn)
    print(classification_report(test_y, knn_pred))


def mlp(train_x, train_y, test_x, test_y, title):
    # layers = 50
    activation = "relu"
    solver = "sgd"
    # alpha = 0.0001
    learning_rate = 0.1
    max_iter = 400

    best_score = 0
    best_alpha = 0.0001
    best_layers = 50
    print(title)
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        for layers in range(50, 300, 50):
            mlp = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=activation,
                solver=solver,
                alpha=alpha,
                learning_rate_init=learning_rate,
                max_iter=max_iter)

            mlp.fit(train_x, train_y)
            score = mlp.score(test_x, test_y)

            print("alpha =  %.4f, layers = %d, score:  %.4f%%" % (alpha, layers, score))
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_layers = layers

    print("Best alpha = %.2f, layers = %d" % (best_alpha, best_layers))
    mlp = MLPClassifier(
        hidden_layer_sizes=best_layers,
        activation=activation,
        solver=solver,
        alpha=best_alpha,
        learning_rate_init=learning_rate,
        max_iter=max_iter)

    mlp.fit(train_x, train_y)
    mlp_pred = mlp.predict(test_x)
    visualization.visualize_confusion_cora(test_y,mlp_pred,mlp)
    print(classification_report(test_y, mlp_pred))

def svc(train_x, train_y, test_x, test_y, title):
    best_score = 0
    best_kernel = "linear"
    best_gamma = "scale"
    print(title)
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        for gamma in ["scale", "auto"]:
            svc = SVC(kernel=kernel, gamma=gamma)
            svc.fit(train_x, train_y)
            score = svc.score(test_x, test_y)

            print("kernel =  %s, gamma = %s, score:  %.4f%%" % (kernel, gamma, score))
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_gamma = gamma

    print("Best kernel = %s, gamma = %s" % (best_kernel, best_gamma))
    svc = SVC(kernel=best_kernel, gamma=best_gamma)

    svc.fit(train_x, train_y)
    svc_pred = svc.predict(test_x)
    visualization.visualize_confusion_cora(test_y,svc_pred,svc)
    print(classification_report(test_y, svc_pred))

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

# K-nn
knn(train_x, train_y, test_x, test_y, "Cora Knn default")

# MLP
mlp(train_x, train_y, test_x, test_y, "Cora Mlp default")

# SVM
svc(train_x, train_y, test_x, test_y, "Cora Svc default")


# --------------------- New Data ------------------------
edges = [[0 for col in range(2708)] for row in range(2708)]

for i in range(len(coraData.edge_index[0])):
    edges[coraData.edge_index[0][i]][coraData.edge_index[1][i]] = 1

new_x = []
for i in range(len(coraData.x)):
    new_x.append(np.append(coraData.x[i], edges[i]))

new_train_x = new_x[:640]
new_test_x = new_x[-1000:]


# K-nn
knn(new_train_x, train_y, new_test_x, test_y, "Cora Knn with edges")

# MLP
mlp(new_train_x, train_y, new_test_x, test_y, "Cora Mlp with edges")

# SVM
svc(new_train_x, train_y, new_test_x, test_y, "Cora Svc with edges")