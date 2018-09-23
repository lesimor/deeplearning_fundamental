import tensorflow as tf


def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', [784, 10], initializer=init)
    b = tf.get_variable('b', [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output


def loss(output, y):
    dot_product = y * tf.log(output)
    cross_entropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def training(cost, global_step, learning_rate):
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
