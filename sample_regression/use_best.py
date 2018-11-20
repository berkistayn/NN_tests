import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import h5py

# recreate the model
dense_layer = 4
layer_size = 128
model = Sequential()
model.add(Dense(layer_size, input_dim=2))
model.add(Activation('relu'))
for _ in range(dense_layer-1):
    print('added')
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
model.add(Dense(1))
model.load_weights('best_weights.h5')


def dataGen(mean, cov, size):
    X = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)
    return X

def some_func(X):
    return np.multiply(np.sin(X[:, 0]), np.power(np.cos(X[:, 1]), 2))

def plot(X_test, y_predicted, y_true):
    # plot data
    fig = plt.figure(figsize=(8, 6))
    ax2 = fig.add_subplot(111, projection='3d')

    ax2.scatter(X_test[:, 0], X_test[:, 1], y_predicted)
    ax2.scatter(X_test[:, 0], X_test[:, 1], y_true)
    ax2.set_title('Prediction')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')

    plt.show()

# For our example sample regression, we are learning a mathematical func.
# So far we know our model fits the training data as well as the standard test data.
# To test te degree of learning we will try some new test data:
# We want to see if it got an idea of sin, cos, multiply:

# data with non-zero mean:
X = dataGen([3, 3], [[1, 0], [0, 1]], 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)
# data with zero-mean but different standard deviation
X = dataGen([0, 0], [[1, 0], [0, 10]], 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)
# data outside from training space:

X = dataGen([30, 30], [[1, 0], [0, 1]], 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)
# FAILED HARD
