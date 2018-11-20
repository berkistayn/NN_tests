from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras as keras
import random
import datetime

start_time = time.time()
# generate some data:
def some_func(X):
    return np.multiply(np.sin(X[:, 0]), np.power(np.cos(X[:, 1]), 2))

# generate data
size = 1000
mean = [0, 0]
cov = [[2, 0], [0, 2]]
X = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)
y = some_func(X).reshape(size, 1)

# plot data
show_input          = False
show_inputAndOutput = False

if show_input is True:
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1])
    ax.set_title('randomly drawn data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()

if show_inputAndOutput is True:
    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], y)
    ax2.set_title('y = some_func(X1, X2)')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')
    plt.show()

# do a random search among network shapes (better than grid search)
# network shapes:
dense_layers = [8,9,10,11]
layer_sizes = [32]
for _ in range(10):
    dense_layer = random.choice(dense_layers)
    layer_size  = random.choice(layer_sizes)

    startRun_time = str(datetime.datetime.now())
    startRun_time = startRun_time.replace(' ', '__')
    startRun_time = startRun_time.replace(':', '_')
    startRun_time = startRun_time.replace('.', '__')
    NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, startRun_time)
    print(NAME)

    model = Sequential()
    for _ in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))

    # add the final layer
    model.add(Dense(1))

    tensorboard_callb = TensorBoard(log_dir="logs/{}".format(NAME))

    model.compile(loss      = 'mse',
                  optimizer = 'adam',
                  metrics   = ['mae'],
                  )

    model.fit(X, y,
              batch_size       = 32,
              epochs           = 15,
              validation_split = 0.25,
              callbacks=[tensorboard_callb])


print('Time elapsed: ', time.time() - start_time, 'seconds')