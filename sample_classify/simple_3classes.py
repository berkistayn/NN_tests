from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import optimizers
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


# detect 3 classes: numbers greater than M1/ smaller than M2 /between M1 and M2
M1 = 3
M2 = 4

start_time = time.time()
# generate some data:
size = 10000
X = np.random.normal(0, 5, size).reshape(size, 1)

y1 = np.less(X, M1).astype(int).reshape(size, 1)
y2 = np.logical_and(X>=M1, X<=M2).astype(int).reshape(size, 1)
y3 = np.greater(X, M2).astype(int).reshape(size, 1)
y = np.hstack((y1, y2, y3))

y_plot = np.zeros((y.shape[0], 1))
for row in range(y.shape[0]):
    k = np.argmax(y[row, :])
    y_plot[row] = k

# plot data
show_inputAndOutput = False
if show_inputAndOutput is True:
    # plot data
    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(111)
    ax1.scatter(X, y_plot, c=y_plot, s=40, cmap=plt.cm.Spectral)
    plt.show()

# do a random search among network shapes (better than grid search)
# network shapes:
dense_layer = 1
layer_size = 6
epochss    = 1300

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
model.add(Dense(3))
model.add(Activation('softmax'))

tensorboard_callb = TensorBoard(log_dir="logs/{}".format(NAME))
earlystop_cb = EarlyStopping(monitor='categorical_crossentropy', min_delta=0.001, patience=3)

model.compile(loss      = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics   = ['categorical_crossentropy'],
              )

model.fit(X, y,
          batch_size       = 2048,
          epochs           = epochss,
          validation_split = 0.2,
          callbacks=[tensorboard_callb, earlystop_cb])
          #callbacks=[tensorboard_callb])



# generate some TEST data:
size = 1000
X_test = np.random.normal(0, 10, size).reshape(size, 1)
y1_t = np.less(X_test, M1).astype(int).reshape(size, 1)
y2_t = np.logical_and(X_test>=M1, X_test<=M2).astype(int).reshape(size, 1)
y3_t = np.greater(X_test, M2).astype(int).reshape(size, 1)
y_t = np.hstack((y1_t, y2_t, y3_t))

y_t_plot = np.zeros((y_t.shape[0], 1))
for row in range(y_t.shape[0]):
    k = np.argmax(y_t[row, :])
    y_t_plot[row] = k

y_p = model.predict(X_test)

y_p_plot = np.zeros((y_p.shape[0], 1))
for row in range(y_p.shape[0]):
    k = np.argmax(y_p[row, :])
    if y_p[row, k] >= 0.4:
        y_p_plot[row] = k
    else:
        y_p_plot[row] = -1

# plot data
# plot data
fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(X_test, y_t_plot, c=y_t_plot, s=40, cmap=plt.cm.rainbow)
ax2.scatter(X_test, y_p_plot, c=y_t_plot, s=40, cmap=plt.cm.rainbow)

ax2.set_title('Prediction')
ax1.set_title('True')
ax1.set_ylim([-2, 3])
ax2.set_ylim([-2, 3])

plt.show()

model.save_weights('bin_best_weights.h5')


