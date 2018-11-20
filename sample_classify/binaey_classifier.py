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


# detect numbers greater than M
M = -2

start_time = time.time()
# generate some data:
size = 10000
X = np.random.normal(0, 5, size).reshape(size, 1)

y1 = np.greater(X, M).astype(int).reshape(size, 1)
y2 = np.less(X, M).astype(int).reshape(size, 1)
y = np.hstack((y1, y2))

print(y)
# plot data
show_inputAndOutput = True
if show_inputAndOutput is True:
    plt.scatter(X, y1, c=y1, cmap=plt.cm.coolwarm)
    plt.title('generated data')
    plt.show()

# do a random search among network shapes (better than grid search)
# network shapes:
dense_layer = 1
layer_size = 4
epochss    = 8

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
model.add(Dense(2))
model.add(Activation('softmax'))

tensorboard_callb = TensorBoard(log_dir="logs/{}".format(NAME))
earlystop_cb = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2)

model.compile(loss      = 'binary_crossentropy',
              optimizer = 'adam',
              metrics   = ['accuracy'],
              )

model.fit(X, y,
          batch_size       = 4096,
          epochs           = epochss,
          validation_split = 0.2,
          callbacks=[tensorboard_callb, earlystop_cb])

for _ in range(5):
    # generate some TEST data:
    X_test = np.random.normal(0, 10, 1000).reshape(1000, 1)
    y_test = np.greater(X_test, M).astype(int)

    y_pred = model.predict(X_test)

    y_res  = np.arange(0, X_test.shape[0], 1)
    print(y_pred)
    for row in range(y_pred.shape[0]):
        if y_pred[row, 1] <= y_pred[row, 0]:
            y_res[row] = 1
        else:
            y_res[row] = 0

    # plot data
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(X_test, y_test, c=y_test, cmap='viridis')
    ax2.scatter(X_test, y_res, c=y_test, cmap='viridis')

    ax1.set_title('True values')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_title('Prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()

model.save_weights('bin_best_weights.h5')


