from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras as keras


# generate some data:
def some_func(X):
    return np.sin(X[:, 0]) + np.sin(X[:, 1])

# generate data
size = 1000
mean = [0, 0]
cov = [[1, 0], [0, 1]]
X = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)
y = some_func(X).reshape(size, 1)

# plot data
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax.scatter( X[:, 0], X[:, 1], y)
ax2.scatter(X[:, 0], X[:, 1])
ax3.scatter(X[:, 0], y)

ax.set_title('y = some_func(X1, X2)')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax2.set_title('randomly drawn data')
ax2.set_xlabel('X1')
ax2.set_xlabel('X2')
ax3.set_title('Prediction')
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_zlabel('Y')


plt.tight_layout()

dense_layers = [2, 3]
layer_sizes = [8, 16, 32]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:

        NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
        print(NAME)

        model = Sequential()
        for _ in range(dense_layer):
            model.add(Dense(layer_size))
            model.add(Activation('relu'))

        # add the final layer
        model.add(Dense(1))

        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae'],
                      )

        model.fit(X, y,
                  batch_size=32,
                  epochs=20,
                  validation_split=0.2,
                  callbacks=[tensorboard])

plt.show()

