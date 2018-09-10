import tensorflow as tf
import numpy as np

x_data = np.array(
        [[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
y_data = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1]
    ])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2,2],-1.,1.))
W2 = tf.Variable(tf.random_uniform([2,3],-1.,1.))
b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([3]))
W3 = tf.Variable(tf.random_uniform([2,2],-1.,1.))
b3 = tf.Variable(tf.zeros([2]))
L1 = tf.add(tf.matmul(X,W1),b1)
L1 = tf.nn.relu(L1)
print(L1)
L2 = tf.add(tf.matmul(L1,W3),b3)
model = tf.add(tf.matmul(L2,W2),b2)
print(model)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    if (step+1)%10==0:
        print(step+1,sess.run(cost,feed_dict={X:x_data,Y:y_data}))

prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)
print('예측 ',sess.run(prediction,feed_dict={X:x_data}))
print('실제 ',sess.run(target,feed_dict={Y:y_data}))

is_correct = tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy*100, feed_dict={X:x_data,Y:y_data}))
