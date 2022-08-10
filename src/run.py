import os
import numpy as np
import tensorflow as tf

from model import Link3Vec

def train(config):
    model = Link3Vec(config=config)
    for x in range(model.startIndex , config['iteration']):
        train = tf.random.shuffle(config['train'], seed=None, name=None)
        tensor = []
        _tSize = 5000#train.shape[0]
        for tIndex in range(0 , _tSize):
            triple = tf.gather(train, tIndex)
            # if x%3 ==0 :
            tensor.append((model._score_and_update_tail(triple, tIndex) + model._score_and_update_rel(triple, tIndex) + model._score_and_update_head(triple, tIndex) ) / 3 )      
            
        if x > 20 and x % 5 == 0:
            model._save(x)
        if x % 3 ==0:
            print(sum(tensor) / _tSize)

def run():
    config = {}
    
    config['path'] = '../data/WN18_numpy/'
    config['train'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "train.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['validation'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "validation.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['test'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "test.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['relations'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "relations.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['entities'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "entities.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['_save_embedings'] = False
    config['read_Last_state'] = False
    config['iteration'] = tf.constant(2000)
    config['dim'] = tf.constant(100)
    config['kns'] = tf.constant(1)
    config['kns_r'] = tf.constant(1)
    config['alpha'] = tf.Variable(0.07, dtype = tf.float64)
    config['beta'] = tf.Variable(0.007, dtype = tf.float64)

    with tf.device('/cpu:0'):
        config['nSam_ent'] = tf.gather( tf.random.categorical([tf.gather(config['entities'], 2, axis=1)], tf.math.multiply(config['kns'],config['train'].shape[0]) , dtype=None, seed=None),0)
        config['nSam_rel'] = tf.gather( tf.random.categorical([tf.gather(config['relations'], 2, axis=1)], tf.math.multiply(config['kns_r'],config['train'].shape[0]) , dtype=None, seed=None),0)

    train(config)

if __name__ == "__main__":
    run()    