import tensorflow as tf
from Z_function_practice.functions import show_operations

c1 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
v1 = tf.Variable([[1, 2, 3], [7, 8, 9]])

print('-----------reshape------------')
RESHAPE = [[2, -1], [-1, 3], [-1, 3, 4, 1], [-1]]
show_operations(tf.reshape(c1, [2, -1]))  # [[1 3 5 7 9 0] [2 4 6 8 3 7]]
show_operations(tf.reshape(c1, [-1, 3]))  # [[1 3 5] [7 9 0] [2 4 6] [8 3 7]]
show_operations(tf.reshape(c1, [-1, 3, 4, 1])) # [[[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]]]]
show_operations(tf.reshape(v1, [-1]))  # [1 2 3 7 8 9]

c2 = tf.reshape(c1, [2, 2, 1, 3])
c3 = tf.reshape(c1, [1, 4, 1, 3, 1])

print('-----------squeeze------------')  # reemoves dimensions of size 1
# [[[[1 3 5]] [[7 9 0]]]  [[[2 4 6]] [[8 3 7]]]]
show_operations(c2)
show_operations(tf.squeeze(c2))  # [[[1 3 5] [7 9 0]]  [[2 4 6] [8 3 7]]]

# [[[[[1] [3] [5]]]  [[[7] [9] [0]]]  [[[2] [4] [6]]]  [[[8] [3] [7]]]]]
show_operations(c3)
show_operations(tf.squeeze(c3))  # [[1 3 5] [7 9 0] [2 4 6] [8 3 7]]
