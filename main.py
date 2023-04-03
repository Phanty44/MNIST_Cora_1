from sklearn import datasets

import visualization
import feature_extr
import MNIST_KNN, MNIST_MLP

from sklearn.model_selection import train_test_split

x, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
x = x / 255.0
x = x.to_numpy()
y = y.to_numpy()

(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=10000, random_state=84, shuffle=True)
# uniform filter, pooling
(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.1, random_state=84)
# Extract HOG features from the training data

visualization.visualize_hog(x_train)
visualization.visualize_maxpool(x_train)

kVals = range(1, 10, 1)

hog_features_train, hog_features_test, hog_features_eval = feature_extr.hog_features_extr(x_train, x_test, x_val)
x_train_pooled, x_test_pooled, x_val_pooled = feature_extr.mp_feature_extr(x_train, x_test, x_val)

# MNIST_KNN.knn_hog(hog_features_train, y_train, hog_features_eval, y_val, hog_features_test, y_test, kVals)
# MNIST_KNN.knn_mp(x_train_pooled, y_train, x_val_pooled, y_val, x_test_pooled, y_test, kVals)
MNIST_MLP.mlp(hog_features_train, y_train, hog_features_eval, y_val, hog_features_test, y_test)
MNIST_MLP.mlp(x_train_pooled, y_train, x_val_pooled, y_val, x_test_pooled, y_test)
