from sklearn.neural_network import MLPClassifier


def mlp(x_train, y_train, x_val, y_val, x_test, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(60,),
        max_iter=10,
        alpha=1e-4,
        solver="sgd",
        verbose=10,
        random_state=1,
        learning_rate_init=0.2,
    )
    model.fit(x_train, y_train)
    print("Training set score: %f" % model.score(x_train, y_train))
    print("Test set score: %f" % model.score(x_test, y_test))


