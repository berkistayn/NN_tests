from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import pickle
import datetime
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# define sin(x) as a custom activation
def sinu(x):
    return K.sin(x)

def cosu(x):
    return K.cos(x)

get_custom_objects().update({'sinu': Activation(sinu)})
get_custom_objects().update({'cosu': Activation(cosu)})

# generate some data:
def some_func(X):
    return np.sin(X)

# generate data
size = 10000
mean = 0
cov = 3
X = np.random.normal(mean, cov, size).reshape(size, 1)
y = some_func(X).reshape(size, 1)

# set parameters according to search results
dense_layer = 2
layer_size = 8
epochss = 100

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

tensorboard = TensorBoard(log_dir="best/{}".format(NAME))

optim = optimizers.Adam()
model.compile(loss       = 'mse',
              optimizer  = optim,
              metrics    = ['mae'],
              )

model.fit(X, y,
          batch_size       = 256,
          epochs           = epochss,
          validation_split = 0.2,
          callbacks        = [tensorboard]
          )

# test the model
test_size = 1000
X_test = np.random.normal(mean, cov, test_size).reshape(test_size, 1)
y_predicted = model.predict(X_test)
y_true = some_func(X_test)

# plot data
fig = plt.figure(figsize=(8, 6))
ax2 = fig.add_subplot(111)

ax2.scatter(X_test, y_predicted)
ax2.scatter(X_test, y_true)
ax2.set_title('Prediction')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.show()

model.save_weights('best_weights.h5')


