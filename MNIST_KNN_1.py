import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def mnist_knn_1(x_train, y_train, x_val, y_val, x_test, y_test):
    (x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train,
                                                        test_size=0.1, random_state=84)

    kVals = range(1, 5, 1)
    accuracies = []

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in range(1, 5, 1):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)

        # evaluate the model and update the accuracies list
        score = model.score(x_val, y_val)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

        predictions = model.predict(x_test)
        print(classification_report(y_test, predictions))
    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))
