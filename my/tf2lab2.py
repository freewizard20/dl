import tensorflow as tf

a = tf.Variable([2,2])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(a))