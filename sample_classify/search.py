from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import optimizers
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import datetime
import random


start_time = time.time()
# generate some data:
N = 198 # number of points per class
D = 2 # dimensionality
no_classes = 3 # number of classes
X = np.zeros((N * no_classes, D)) # data matrix (each row = single example)
y = np.zeros([N * no_classes, no_classes], dtype='uint8') # class labels
y_plot = np.zeros(N*no_classes, dtype='uint8')
start_ind = 0
for j in range(0, no_classes):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[range(start_ind, N*(j+1)), j] = 1
    start_ind = start_ind + N
    y_plot[ix] = j

# shuffle it
c = np.hstack((X, y))
np.random.shuffle(c)
X = c[:, 0:D]
y = c[:, D:None]


# do a random search among network shapes (better than grid search-sometimes-)
# network shapes:
epochss = 100
dense_layers = [0, 1, 2, 3]
layer_sizes = [2, 4, 8, 16]
for _ in range(30):
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
        model.add(Activation('tanh'))

    # add the final layer
    model.add(Dense(no_classes))
    model.add(Activation('softmax'))

    tensorboard_callb = TensorBoard(log_dir="logs/{}".format(NAME))
    earlystop_cb = EarlyStopping(monitor='categorical_accuracy', min_delta=0.01, patience=15)

    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics   = ['categorical_accuracy'],
                  )

    model.fit(X, y,
              batch_size       = 32,
              epochs           = epochss,
              validation_split = 0.2,
              #callbacks=[tensorboard_callb, earlystop_cb])
              callbacks=[tensorboard_callb])




