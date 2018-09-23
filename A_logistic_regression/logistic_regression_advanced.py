import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from A_logistic_regression.utils.convolution import *
from A_logistic_regression.utils.generator import *

learning_rate = 0.1
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_ = tf.placeholder(tf.float32, shape=[None, 10])

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# 처음 두 개의 차원은 윈도우의 크기, 세 번째는 입력 채널의 수, 마지막은 출력 채널의 수
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
h_pool1 = max_pool_2x2(h_conv1)  # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
h_pool2 = max_pool_2x2(h_conv2)  # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
