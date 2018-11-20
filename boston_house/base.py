import pandas as pd, xlrd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# build NN:

def build_model(LR = 0.0003, l2 = 0.0001):
    #activation_func = tf.nn.relu
    activation_func = tf.nn.tanh

    model = keras.Sequential([
    keras.layers.Dense(64, activation=activation_func, input_shape=(train_data.shape[1],),
                                                       kernel_regularizer = regularizers.l2(l2)),
    keras.layers.Dense(64, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(20, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(20, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(3, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(LR)
    optimizer = tf.train.AdamOptimizer(LR)
    optimizer = keras.optimizers.Adadelta()
    #optimizer = keras.optimizers.Nadam()

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])

# get data:

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# build NN:

model = build_model()
# add regularization:
# model.add(Dropout(0.2))
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 500

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

plot_history(history)
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\nTesting set Mean Abs Error: {:7.2f}".format(mae))

plt.show()

#

plt.clf()
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()