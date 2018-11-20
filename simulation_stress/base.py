import pandas, xlrd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

class dataReader:
    def __init__(self):
        self.file_path = None
        self.df        = None
        self.data      = None
        self.train     = None
        self.cv        = None
        self.test      = None
        self.pure_test = None

        self.mean      = None
        self.std       = None

    def setPath(self, file_path):
        self.file_path = file_path

    def read_excel(self, no_data, no_in, no_out,
                   index_out_Col_start, index_startRow=0, index_in_Col_start=0):
        # get data from excel file
        self.df = pandas.read_excel(self.file_path)
        data_in = np.zeros([no_data, no_in])
        df_num = self.df.values
        for i in range(index_in_Col_start, index_in_Col_start + no_in):
            data_in[:, i - index_in_Col_start] = df_num[index_startRow:None, i]
        data_out = np.zeros([no_data, no_out])
        for i in range(index_out_Col_start, index_out_Col_start + no_out):
            data_out[:, i - index_out_Col_start] = df_num[index_startRow:None, i]
        data = np.hstack((data_in, data_out))
        np.random.shuffle(data)
        self.data = data

    def prepareData(self, ratio_train2all):
        # split data into train, test, cross-validation(cv):
        ax1 = int(round(self.data.shape[0] * ratio_train2all))
        ax2 = ax1 + int(round(self.data.shape[0] * (1 - ratio_train2all) / 2))
        ax3 = None  # (to end)
        train = self.data[0:ax1,   :]
        cv    = self.data[ax1:ax2, :]
        self.pure_test  = self.data[ax2:ax3, :]

        # pre-process data, normalization:
        # do not use info except for training data
        self.mean = train.mean(axis=0)
        self.std  = train.std(axis=0)

        self.train = (train - self.mean) / self.std
        self.cv    = (cv - self.mean) / self.std
        self.test  = (self.pure_test - self.mean) / self.std

    def getData(self):
        return self.train, self.cv, self.test

    def getTest(self):
        return self.pure_test


# build NN:

def build_model(LR = 0.0003, l2 = 0.0001):
    #activation_func = tf.nn.relu
    activation_func = tf.nn.tanh

    model = keras.Sequential([
    keras.layers.Dense(12, activation=activation_func, input_shape=(train_data.shape[1],),
                                                       kernel_regularizer = regularizers.l2(l2)),
    keras.layers.Dense(6, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(20, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(20, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    #keras.layers.Dense(3, activation=activation_func, kernel_regularizer=regularizers.l2(l2)),
    keras.layers.Dense(2)
    ])

    optimizer = tf.train.RMSPropOptimizer(LR)
    optimizer = tf.train.AdamOptimizer(LR)
    optimizer = keras.optimizers.Adadelta()
    optimizer = keras.optimizers.Nadam()

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
  plt.ylim([0, 1])


dR = dataReader()
dR.setPath('C:/Users/tosun/Desktop/DATA_for_ANN.xlsx')
dR.read_excel(no_data=120, no_in=10, no_out=2, index_out_Col_start=11, index_startRow=1)
dR.prepareData(ratio_train2all=2/3)
# get normalized data from the excel file
train, cv, test = dR.getData()

keras_data = np.vstack((train,cv))
train_data   = keras_data[:, 0:10]
train_labels = keras_data[:, 10:12]
test_data   = test[:, 0:10]
test_labels = test[:, 10:12]


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

EPOCHS = 1000

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.15, verbose=0,
                    callbacks=[PrintDot()])

plot_history(history)
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\nTesting set Mean Abs Error: {:7.2f}".format(mae))

plt.show()

#

plt.clf()
#kkk = dR.getTest()
#test_labels = kkk[:, 10:12]
#test = kkk[:, 0:10]

test_predictions = model.predict(test_data).flatten()
#test_predictions = model.predict(test).flatten()
print('----')
print(model.predict(test_data))
#print(model.predict(test))
print(test_labels)


plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()