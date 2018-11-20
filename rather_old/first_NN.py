import numpy as np
import tensorflow as tf
from tensorflow import keras
import time, os
from matplotlib import pyplot as plt

print(tf.VERSION)
print(keras.__version__)

start_time = time.time()
w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])
coefficients = np.array([[1.], [-10.], [25.]])

cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
#cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

no_iterations = 200
for i in range(no_iterations):
    session.run(train, feed_dict={x:coefficients})

writer = tf.summary.FileWriter(r'tf_dat')
writer.add_graph(session.graph)
writer.close()

print('polynomial coefficients:\n', coefficients)
print('result:', session.run(w))
print('time elapsed:', time.time() - start_time, ' seconds.')

