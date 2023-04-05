from sklearn.metrics import classification_report
from sklearn.svm import SVC

import visualization


def mnist_svm(x_train, y_train, x_val, y_val, x_test, y_test, prime_dataset):
    kernel = ["linear"]
    for i in kernel:
        model = SVC(kernel=i, gamma="auto", degree=3)
        model.fit(x_train, y_train)

        print("Training set score: %f" % model.score(x_train, y_train))
        print("Validation set score: %f" % model.score(x_val, y_val))
        print("Test set score: %f" % model.score(x_test, y_test))
        predictions = model.predict(x_test)
        #print(classification_report(y_test, predictions))

        # visualization.visualize_wrong(predictions, y_test, prime_dataset)
