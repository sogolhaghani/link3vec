import os
import numpy as np
import tensorflow as tf

from model import Link3Vec

def train(config):
    model = Link3Vec(config=config)
    for x in range(config.startIndex , config.iteration):
        train = tf.random.shuffle(config.train, seed=None, name=None)
        for tIndex in range(0 , train.shape[0]):
            triple = tf.gather(train, tIndex)
            model._score_and_update(triple, tIndex)
            
        if x > 20 and x % 5 == 0:
            model._save(x) 

def run():
    config = {}
    
    config.path = '../data/WN18_numpy'
    config.train = tf.convert_to_tensor( np.load(os.path.join(config.path, "train.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config.validation = tf.convert_to_tensor( np.load(os.path.join(config.path, "validation.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config.test = tf.convert_to_tensor( np.load(os.path.join(config.path, "test.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config.relations = tf.convert_to_tensor( np.load(os.path.join(config.path, "relations.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config.entities = tf.convert_to_tensor( np.load(os.path.join(config.path, "entities.npy") ).astype(dtype=np.float64), dtype=tf.float64)

    config.save_embedings = False
    config.read_Last_state = False
    config.iteration = tf.constant(2000)
    config.dim = tf.constant(100)
    config.kns = tf.constant(1)
    config.kns_r = tf.constant(1)
    config.alpha = tf.Variable(0.07, dtype = tf.float64)
    config.beta = tf.Variable(0.007, dtype = tf.float64)

    train(config)