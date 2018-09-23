import tensorflow as tf
from .modules import *

learning_rate = 0.01
training_epoch = 1000
batch_size = 100
display_step = 1

with tf.Graph().as_default():
    x = tf.placeholder('float', [None, 28*28])
    y = tf.placeholder('float', [None, 10])

    output = inference(x)

    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step, learning_rate)

    eval_op = evaluate(output, y)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('./logistic_logs/', graph_def=sess.graph_def)

    init_op = tf.initialize_all_variables()

    sess.run(init_op)