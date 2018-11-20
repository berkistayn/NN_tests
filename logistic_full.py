import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# final state(12.11.2018): fails to classify data in two groups, rather it splits
# them by a line which minimizes the distance between the line and ALL data points.
# Seems like a mistake exists in the definition of error func. or gradient.

# create the data sets c1, c2
# which are NOT linearly separable

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2*sigmoid(2*x) - 1

start_time = time.time()


size = 500

mean = [5, 5]
cov = [[1, 0], [0, 5]]
c1a = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)

mean = [-5, -5]
cov = [[5, 0], [0, 1]]
c1b = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)

mean = [-6.5, 1]
cov = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(mean, cov, size).reshape(size, 2)

c1 = np.concatenate((c1a, c1b), axis=0).reshape(2*size, 2)
np.random.shuffle(c1)

x = np.concatenate((c1, c2)).T
y1 = np.zeros([2*size, 1])
y2 = np.ones([size, 1])
y = np.concatenate((y1, y2)).T
w = np.random.uniform(-0.5, 0.5, size=[2, 1]).reshape(2, 1)
b = 1

no_iter = 1000
err_stor = np.zeros([no_iter, 1])
for i in range(no_iter):
    LR = 0.01
    z = np.dot(np.transpose(w), x) + b
    A = tanh(z)
    dZ = A - y
    dW = 1/(3*size) * np.dot(x, dZ.T)
    dB = 1/(3*size) * np.sum(dZ)
    w = w - LR * dW
    b = b - LR * dB
    err_stor[i] = np.mean(np.power(-1/2*(A - y), 2))

    if i % 100 == 0:
        # PLOT :
        plt.clf()
        lin_x = np.arange(-14, 14, 0.005).reshape(2, 2800)
        lin = np.arange(-14, 14, 0.01).reshape(2800, 1)
        plt.plot(lin, np.dot(w.T, lin_x).T + b, color='black')
        plt.scatter(c1[:, 0], c1[:, 1])
        plt.scatter(c2[:, 0], c2[:, 1])
        plt.ylim(-10, 10)
        plt.xlim(-15, 15)
        plt.pause(0.2)

print('out!')
plt.show()
iter = np.arange(1, no_iter + 1, 1).reshape(no_iter, 1)
plt.plot(iter, err_stor)
plt.show()
print('time elapsed:', time.time() - start_time)


'''
lin_x = np.arange(-14, 14, 0.005).reshape(2, 2800)
lin = np.arange(-14, 14, 0.01).reshape(2800, 1)
plt.plot(lin, np.dot(w.T, lin_x).T + b, color='black')
plt.scatter(c1[:, 0], c1[:, 1])
plt.scatter(c2[:, 0], c2[:, 1])
plt.show()
'''

'''
# create graph:
with tf.name_scope('init'):
    w = tf.Variable(np.random.rand(2, 1).reshape(2, 1), dtype=tf.float32)
    #w = tf.Variable(np.array([-0.5, 0.4]).reshape(2, 1), dtype=tf.float32)
    b = tf.constant(1, dtype=tf.float32)
    x = tf.placeholder(tf.float32, [2, 1])
    y = tf.placeholder(tf.float32, [1, 1])
    cost = -1/2 * (y - tf.sigmoid(tf.matmul(tf.transpose(w), x) + b))**2
    tf.summary.histogram('cost', cost)
    tf.summary.histogram('weights', w)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
'''


