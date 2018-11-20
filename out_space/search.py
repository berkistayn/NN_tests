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
    return np.sin(X)

# generate data
size = 1000
mean = 0
cov = 2
X = np.random.normal(mean, cov, size).reshape(size, 1)
y = some_func(X).reshape(size, 1)

# plot data
show_inputAndOutput = True

if show_inputAndOutput is True:
    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(X, y)
    ax2.set_title('y = sin(X)')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    plt.show()

# do a random search among network shapes (better than grid search)
# network shapes:
dense_layers = [0, 1, 2]
layer_sizes = [2, 4, 8]
for _ in range(20):
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
              batch_size       = 128,
              epochs           = 15,
              validation_split = 0.25,
              callbacks=[tensorboard_callb])


print('Time elapsed: ', time.time() - start_time, 'seconds')