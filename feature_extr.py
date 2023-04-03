import numpy as np
from skimage.feature import hog
from skimage.measure import block_reduce


def hog_features_extr(x_train, x_test, x_val):
    hog_features_train = []
    for i in range(x_train.shape[0]):
        hog_features_train.append(
            hog(x_train[i].reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))

    # Extract HOG features from the testing data
    hog_features_test = []
    for i in range(x_test.shape[0]):
        hog_features_test.append(
            hog(x_test[i].reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))

    hog_features_eval = []
    for i in range(x_val.shape[0]):
        hog_features_eval.append(
            hog(x_val[i].reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))

    return hog_features_train, hog_features_test, hog_features_eval


def mp_feature_extr(x_train, x_test, x_val):
    # Perform 2x2 max pooling on 28x28 training and testing images
    x_train_pooled = block_reduce(x_train.reshape(-1, 28, 28), block_size=(1, 2, 2), func=np.max)
    x_val_pooled = block_reduce(x_val.reshape(-1, 28, 28), block_size=(1, 2, 2), func=np.max)
    x_test_pooled = block_reduce(x_test.reshape(-1, 28, 28), block_size=(1, 2, 2), func=np.max)

    # Flatten the pooled training and testing data to be used in a KNN classifier
    x_train_pooled = x_train_pooled.reshape(-1, 196)
    x_val_pooled = x_val_pooled.reshape(-1, 196)
    x_test_pooled = x_test_pooled.reshape(-1, 196)

    return x_train_pooled, x_test_pooled, x_val_pooled
