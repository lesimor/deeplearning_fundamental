import tensorflow as tf


def show_operations(t):
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    print('eval :', t.eval())
    print('shape :', tf.shape(t))
    print('size  :', tf.size(t))
    print('rank  :', tf.rank(t))
    print('get shape :', t.get_shape())
    print('=========' * 10)

    sess.close()
