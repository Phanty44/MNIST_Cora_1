from scipy.ndimage import uniform_filter
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

x, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
x = x/255.0
x = x.to_numpy()
y = y.to_numpy()
x_sparse = coo_matrix(x)
x, x_sparse, y = shuffle(x, x_sparse, y, random_state=0)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

x_train = MaxPool()
#uniform filter, pooling

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



