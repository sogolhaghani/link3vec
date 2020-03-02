import tensorflow as tf

import random

class TableForNegativeSamples:
    def __init__(self, vocab):
        power =tf.constant( 0.75)
        norm = tf.reduce_sum(tf.pow(tf.cast(vocab, tf.float32), power))  # Normalizing constants
        p = tf.math.truediv( tf.math.pow(tf.cast(vocab, tf.float32), power), norm, name='Cumulative probability')
        self.table_size = tf.constant(100000000, dtype=tf.float32)   
        x = tf.cast(tf.math.multiply(p , self.table_size) , tf.int32)
        # TODO : Refactor
        self.table = tf.Variable(tf.nest.flatten([[i for x in range(x[i])] for i in range(x.get_shape()[0])]))
   
    def sample(self, count):
        indices = tf.random.uniform([1, count], minval=0, maxval=tf.cast(self.table_size, tf.int32), dtype=tf.int32)
        return self.table.numpy()[indices]