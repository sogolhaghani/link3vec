
import tensorflow as tf
i = tf.constant(1)
axis = tf.constant(1)
tensors = tf.Variable([1,5,8,6,3,8,5] , dtype=tf.dtypes.int32)
# tensors = [1,5,8,6,3,8,5]
# tf.concat([tf.expand_dims(t, 0) for t in tensors], 0)
# print(tf.concat([tf.expand_dims(t, 0) for t in tensors], 0))

# z = tf.Variable([lambda i:tensors[i]])
# print(tf.Variable(tf.nest.flatten([[i for x in range(tensors[i])] for i in range(tensors.get_shape()[0])])))

x = tf.random.uniform([1, 10], minval=1, maxval=10, dtype=tf.int32)
print(x)
