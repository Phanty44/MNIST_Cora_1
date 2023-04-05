from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

import visualization


def mlp(x_train, y_train, x_val, y_val, x_test, y_test, prime_dataset):
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        batch_size=100,
        max_iter=20,
        alpha=1e-4,
        solver="sgd",
        verbose=10,
        random_state=1,
        learning_rate_init=0.2,
    )
    model.fit(x_train, y_train)
    print("Training set score: %f" % model.score(x_train, y_train))
    print("Validation set score: %f" % model.score(x_val, y_val))
    print("Test set score: %f" % model.score(x_test, y_test))
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    # visualization.visualize_wrong(predictions, y_test, prime_dataset)
