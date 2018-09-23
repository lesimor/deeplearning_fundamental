import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from A_logistic_regression.utils.steps import inference, loss, training, evaluate

learning_rate = 0.1
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
global_step = tf.Variable(0, name='global_step', trainable=False)

x = tf.placeholder(tf.float32, [None, 784])
y = inference(x)
y_ = tf.placeholder(tf.float32, [None, 10])
cost_op = loss(y, y_)

train_op = training(cost_op, global_step, learning_rate)
eval_op = evaluate(y, y_)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})

test_feed_dict = {
    x: mnist.test.images,
    y_: mnist.test.labels
}

accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
print('Accuracy: ', accuracy)
