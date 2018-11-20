import tensorflow as tf

tf.reset_default_graph()

g = tf.Graph()

with g.as_default():
  y = tf.Variable(1)
  tf.summary.scalar('thing', y)
  initialize = tf.global_variables_initializer()

sess = tf.InteractiveSession(graph=g)
sess.run(initialize)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(r'D:tb/sc2', g)

for i in range(10):
    summary = sess.run(merged)
    writer.add_summary(summary, i)

sess.close()