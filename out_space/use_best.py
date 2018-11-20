import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# define sin(x) as a custom activation
def sinu(x):
    return K.sin(x)

get_custom_objects().update({'sinu': Activation(sinu)})

# recreate the model
dense_layer = 2
layer_size = 8

model = Sequential()
model.add(Dense(layer_size, input_dim=1))  # always add input_dim to overcome a bug
model.add(Activation('relu'))
for _ in range(dense_layer-1):
    print('added')
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
model.add(Dense(1))
model.load_weights('best_weights.h5')


def dataGen(mean, cov, size):
    X = np.random.normal(mean, cov, size).reshape(size, 1)
    return X

def some_func(X):
    return np.sin(X)

def plot(X_test, y_predicted, y_true):
    # plot data
    fig = plt.figure(figsize=(8, 6))
    ax2 = fig.add_subplot(111)

    ax2.scatter(X_test, y_predicted)
    ax2.scatter(X_test, y_true)
    ax2.set_title('Prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()

# For our example sample regression, we are learning a mathematical func.
# So far we know our model fits the training data as well as the standard test data.
# To test te degree of learning we will try some new test data:
# We want to see if it got an idea of sin, cos, multiply:

# check the build
X = dataGen(0, 1, 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)

# data with non-zero mean:
X = dataGen(3, 1, 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)

# data with zero-mean but different standard deviation
X = dataGen(0, 5, 1000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)

# data outside from training space:
X = dataGen(30, 20, 20000)

y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)
# fails

# show the output space:

X = np.arange(-20, 20, 0.01).reshape(4000, 1)
y_true = some_func(X)
y_predict = model.predict(X)

plot(X, y_predict, y_true)


