import tensorflow as tf
import numpy as np

hello = tf.constant('hello world')

a = tf.constant(np.array([1,2,3]))
b = tf.constant(np.array([2,3,4]))
c = tf.add(a,b)

sess = tf.Session()
print(sess.run([a,b,c]))