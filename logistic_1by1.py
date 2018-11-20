import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt

# works as stochastic GD: improves considering the fed vector only. Thus it gets
# stuck in a certain local optimum(perpendicular to desired optimum) time to time.
# (It occurs due to probability distribution, when lines reach the most populated areas.)
# (If it was using batch GD, it would generate very high errors, but with stochastic we get very low errors.)

start_time = time.time()

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

#assign_op = w.assign(np.random.rand(2, 1).reshape(2, 1))
#sess.run(assign_op)
#plt.close()

# call tensorboard:
merged_sum = tf.summary.merge_all()

run_no = 'run' + str(1)
writer = tf.summary.FileWriter(r'D:tb/' + run_no)
writer.add_graph(sess.graph)
# iterations:
no_it = 2000
gen_xs = np.zeros([no_it, 3]).reshape(no_it, 3)
gen_xs[:, 2] = 77
generated_x = np.zeros([2, 1]).reshape(2, 1)
for i in range(no_it):
    # data generation parameters:
    cent1 = 5
    cent2 = -5
    sd1 = 0.5
    sd2  = 1.3
    sd3  = 4

    # generate inputs:
    generated_y = np.array([np.random.randint(0, 2)]).reshape(1, 1)
    if generated_y == 1:
        mean = cent1
        dev0 = sd3
        dev1 = sd1
    elif generated_y == 0:
        mean = cent2
        dev0 = sd3
        dev1 = sd3
    generated_x[0][0] = np.random.normal(mean, dev0)
    generated_x[1][0] = np.random.normal(mean, dev1)

    # store stuff:
    gen_xs[i, 0:2] = np.transpose(generated_x)
    gen_xs[i, 2] = generated_y.copy()
    if i % 5 == 0:
        s = sess.run(merged_sum, feed_dict={x:generated_x, y:generated_y})
        writer.add_summary(s, i)
    sess.run(train, feed_dict={x:generated_x, y:generated_y})

    if i % 100 == 0:
        # PLOT :
        area1 = np.ma.masked_where(gen_xs[:, 2] == 0, np.ones(no_it))
        area2 = np.ma.masked_where(gen_xs[:, 2] == 1, np.ones(no_it))
        plt.scatter(gen_xs[:, 0], gen_xs[:, 1], s=area1, marker='^')
        plt.scatter(gen_xs[:, 0], gen_xs[:, 1], s=area2, marker='o')
        lin_x1 = np.arange(-7, 7, 0.01).reshape(1400, 1)
        lin_x2 = np.arange(-7, 7, 0.01).reshape(1400, 1)
        plt.plot(sess.run(w)[0] * lin_x1 + sess.run(w)[1] * lin_x2 + sess.run(b), lin_x1, color=(0, 0.4 + 0.6*i/no_it, 0))
        plt.ylim(min(gen_xs[:, 1]), max(gen_xs[:, 1]))
        plt.xlim(min(gen_xs[:, 0]), max(gen_xs[:, 0]))
        plt.pause(100/no_it)

plt.show()
writer.close()


print('time elapsed:', time.time() - start_time, 'seconds.')

