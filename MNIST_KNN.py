import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# loop over various values of `k` for the k-Nearest Neighbor classifier

def knn_hog(x_train, y_train, x_val, y_val, x_test, y_test, kVals):
    accuracies = []
    for k in kVals:
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
    print("k=%d with HOG achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                                    accuracies[i] * 100))


def knn_mp(x_train, y_train, x_val, y_val, x_test, y_test, kVals):
    accuracies = []
    for k in kVals:
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
    print("k=%d with MaxPool achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                                        accuracies[i] * 100))
