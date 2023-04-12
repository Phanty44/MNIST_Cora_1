from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import visualization


def mlp(x_train, y_train, x_val, y_val, x_test, y_test, prime_dataset):
    # 'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    # 'alpha': [0.0001, 0.001, 0.01],
    # 'learning_rate_init': [0.001, 0.01, 0.1],
    # 'solver': ["lbfgs","sgd", "adam"],
    # 'activation': ['identity', 'logistic', 'tanh', 'relu']
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'solver': ["lbfgs", "sgd", "adam"],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        learning_rate_init=0.1,
        solver='sgd',
        activation='relu',
        batch_size=100,
        max_iter=40,
        verbose=10,
        random_state=1,
    )
    # grid_search = GridSearchCV(model, param_grid, cv=3)
    # grid_search.fit(x_train,y_train)
    # grid_search.score(x_test,y_test)
    # print('Best hyperparameters:', grid_search.best_params_)

    model.fit(x_train, y_train)
    model.score(x_test, y_test)
    # predict results for test dataset

    print("Training set score: %f" % model.score(x_train, y_train))
    print("Validation set score: %f" % model.score(x_val, y_val))
    print("Test set score: %f" % model.score(x_test, y_test))

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))

    # visualization.visualize_confusion(y_test, predictions, model)
    # visualization.visualize_wrong(predictions, y_test, prime_dataset)
