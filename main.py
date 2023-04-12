import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as datas

import MNIST_KNN
import MNIST_MLP
import MNIST_SVM
import feature_extr
import visualization

x_train = []
x_test = []
y_train = []
y_test = []

trainset = datas.MNIST('~/.pytorch/MNIST_data/', download=True, train=True)
x_train = np.array(trainset.data)
x_train = np.array(x_train, dtype="float") / 255
y_train = np.array(trainset.targets)

testset = datas.MNIST('~/.pytorch/MNIST_data/', download=True, train=False)
x_test = np.array(testset.data)
x_test = np.array(x_test, dtype="float") / 255
y_test = np.array(testset.targets)

(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=84)
# Extract HOG features from the training data

#visualization.visualize_hog(x_train)
#visualization.visualize_maxpool(x_train)

kVals = range(1, 11, 1)

hog_features_train, hog_features_test, hog_features_eval = feature_extr.hog_features_extr(x_train, x_test, x_val)
x_train_pooled, x_test_pooled, x_val_pooled = feature_extr.mp_feature_extr(x_train, x_test, x_val)
print("Model KNN:")
#MNIST_KNN.knn_hog(hog_features_train, y_train, hog_features_eval, y_val, hog_features_test, y_test, kVals, x_test)
#MNIST_KNN.knn_mp(x_train_pooled, y_train, x_val_pooled, y_val, x_test_pooled, y_test, kVals, x_test)
print("Model MLP:")
#MNIST_MLP.mlp(hog_features_train, y_train, hog_features_eval, y_val, hog_features_test, y_test, x_test)
#MNIST_MLP.mlp(x_train_pooled, y_train, x_val_pooled, y_val, x_test_pooled, y_test, x_test)
print("Model SVM:")
MNIST_SVM.mnist_svm(hog_features_train, y_train, hog_features_eval, y_val, hog_features_test, y_test, x_test)
MNIST_SVM.mnist_svm(x_train_pooled, y_train, x_val_pooled, y_val, x_test_pooled, y_test, x_test)
