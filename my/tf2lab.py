import tensorflow as tf

#a = tf.Variable(tf.random_normal([2,2]))
#b = tf.Variable(tf.random_normal([2,1]))
a = tf.constant([[1,1],[1,1]])
b = tf.constant([2,2])

c = a+b

sess = tf.Session()
#sess.run(tf.global_variables_initializer())
print(sess.run(c))