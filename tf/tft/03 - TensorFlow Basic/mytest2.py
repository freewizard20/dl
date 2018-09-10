import tensorflow as tf

X = tf.placeholder(tf.float32, [None,None])
x_data = [[1,2,4],[3,4,1],[5,6,2]]
W=tf.Variable(tf.random_normal([3,3],mean=100,stddev=1))
b = tf.Variable(tf.random_normal([3,3],mean=10,stddev=12))
expr = tf.matmul(X,W)+b
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(expr,feed_dict={X:x_data}))
sess.close()
