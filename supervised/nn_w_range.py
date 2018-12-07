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


# Creates a standard NN ~ one with fully connected layers:
#
#   Input:  number of grids - which are neighbours of current grid.
#           Inputs are defined by range which is the checked cell range.
#           Inputs are defined as a vector:
#
#               for ex: if range = 1;
#                   # : current cell
#                            1                empty
#                          4 # 2         empty  # blocked
#                            3                blocked
#                   Then input vector = {0, 1, 1, 0}
#
#   Output: a vector of 4x1: confidence in {up, down, right, left}

class NN_w_range:
    def __init__(self, range):
        self.range = range

        self.model = None
        self.createModel()

        self.last_output = None
        self.last_move   = None

    def createModel(self, no_dense=2):
        dense_layer = no_dense
        layer_size = int(4 * (self.range * (self.range + 1) / 2))

        self.model = Sequential()
        for _ in range(dense_layer):
            self.model.add(Dense(layer_size))
            self.model.add(Dropout(0.4))
            self.model.add(Activation('relu'))
            #self.model.add(Activation('tanh'))

        # add the final layer
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_crossentropy'],
                      )

    def trainOnce(self, input_vector, label, epochs=1):
        self.model.fit(input_vector, label,
                  batch_size=1,
                  epochs=epochs,
                  validation_split=0)

    def makeMove(self, input_vector):
        self.last_output = self.model.predict(input_vector)[0]
        k = np.argmax(self.last_output)
        move_dict = {0: 'up',
                     1: 'down',
                     2: 'right',
                     3: 'left'}
        self.last_move = move_dict[k]
        print('Confidence in last move: ', self.last_output[k])

        return self.last_move

    def generate_input(self, grid):
        # check the cells defined by range, return blocked_value if wall, 0 if empty.
        # if cell is out of bounds return its value as wall
        curL = grid.currentLoc
        cells = grid.grid
        input_vector = []
        size = int(4 * (self.range * (self.range + 1) / 2))
        print('current Location:    ', curL)

        for ring in range(self.range + 1):
            # start at top, go circular CW
            del_row = -ring
            del_col = 0
            for i in range(ring*4):
                # select the cell
                y = curL[1] + del_col
                x = curL[0] + del_row

                # check the cell value
                if x < 0 or y < 0 or x >= grid.w or y >= grid.h:
                    # out of bounds
                    input_vector.append(1)
                elif cells[x][y] == grid.blocked or cells[x][y] == grid.tail:
                    input_vector.append(1)
                elif cells[x][y] == grid.unblocked:
                    input_vector.append(0)

                # prepare for next cell
                if i < ring:
                    del_col += 1
                    del_row += 1
                if ring <= i < 2* ring:
                    del_col += -1
                    del_row += 1
                if 2*ring <= i < 3* ring:
                    del_col += -1
                    del_row += -1
                if 3 * ring <= i < 4 * ring:
                    del_col += +1
                    del_row += -1

        return np.array(input_vector).reshape(1, size)

    def saveWeights(self, range, allowed_off_steps):
        cur_time = str(datetime.datetime.now())
        cur_time = cur_time.replace(' ', '__')
        cur_time = cur_time.replace(':', '_')
        cur_time = cur_time.replace('.', '__')
        name = "{}-range-{}-off_steps-{}".format(range, allowed_off_steps,
                                                 cur_time)
        self.model.save_weights('weights/' + name + '.h5')
        print('Saved weights:   ', name)

    def loadWeights(self, weights_file):
        self.model.load_weights(weights_file)
        print('Loaded weights:   ', weights_file)


