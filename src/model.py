import os
import numpy as np
import tensorflow as tf

class Link3Vec():
    def __init__(self,config):
        self.train = config.train
        self.test = config.test
        self.validation = config.validation
        self.relations = config.relations
        self.entities = config.entities

        z = tf.constant(5, dtype = tf.float64)
        self.nn1 = tf.random.uniform( shape=(self.entities.shape[0], self.dim),maxval=tf.math.truediv(z , tf.cast(self.dim,tf.float64),'maxval'), dtype=tf.dtypes.float64, seed=1, name='nn1')
        self.nn0 =tf.random.uniform(shape=(self.entities.shape[0], self.dim), maxval=tf.math.truediv(z , tf.cast(self.dim,tf.float64),'maxval'), dtype=tf.dtypes.float64, seed=1, name='nn0')
        self.nn2 = tf.random.uniform(shape=(self.relations.shape[0], self.dim), maxval=tf.math.truediv(z , tf.cast(self.dim,tf.float64),'maxval'), dtype=tf.dtypes.float64, seed=1, name='nn2')     
        self.startIndex=  tf.constant(0)
        if config.read_Last_state == True:
            self.nn0 = tf.convert_to_tensor( np.load(os.path.join(config.path, "nn0.npy") ).astype(dtype=np.float64), dtype=tf.float64)
            self.nn1 = tf.convert_to_tensor( np.load(os.path.join(config.path, "nn1.npy") ).astype(dtype=np.float64), dtype=tf.float64)
            self.nn2 = tf.convert_to_tensor( np.load(os.path.join(config.path, "nn2.npy") ).astype(dtype=np.float64), dtype=tf.float64)
            self.startIndex= np.load(os.path.join(config.path, "x.npy"))
        with tf.device('/cpu:0'):
            self.nSam_ent = tf.gather( tf.random.categorical([tf.gather(self.entities, 2, axis=1)], tf.math.multiply(self.kns,self.train.shape[0]) , dtype=None, seed=None),0)
            self.nSam_rel  = tf.gather( tf.random.categorical([tf.gather(self.relations, 2, axis=1)], tf.math.multiply(self.kns_r,self.train.shape[0]) , dtype=None, seed=None),0)

    def _score_and_update(self, triple, tIndex):
        paddings = tf.constant([[0, 1,], [0, 0]])
        _head_index = tf.gather(triple,0)
        _relation_index = tf.gather(triple,2)
        _vHead = tf.gather(nn0,_head_index)
        _vRel = tf.gather(nn2,_relation_index)

        samples = [tf.slice(self.nSam_ent,begin=[tf.math.multiply(tIndex, self.kns)],size=[self.kns])]
        samples = tf.pad(samples, paddings, constant_values=0)
        samples =  tf.cast(samples, tf.int64)         
        samples = tf.transpose(tf.concat([samples, [[tf.gather(triple,1)],[1]]], 1))
        indices = tf.gather(samples, 0, axis=1)
        _nn1_samples = tf.gather(nn1, indices)
        _sigmoid =tf.math.sigmoid(tf.tensordot( _nn1_samples , tf.transpose(tf.math.add(_vHead , _vRel)) , axes=1))
        cost = tf.math.subtract(tf.cast(tf.gather(samples, 1, axis=1), tf.float64) , _sigmoid)
        g = tf.math.multiply(self.alpha , cost)
        g1 = tf.math.multiply(self.beta , cost)

        # error in first layer
        _nn1_samples_err = tf.math.add(tf.math.multiply(tf.math.add(_vRel,_vHead)  , tf.reshape(g, (tf.math.add(self.kns,1), 1))) , _nn1_samples)
        nn1 = tf.tensor_scatter_nd_update(nn1,tf.expand_dims(indices, 1),_nn1_samples_err)
      
        # error second layer nn2
        err_nn2 = tf.math.multiply( _nn1_samples, tf.reshape(g1, (g1.shape[0], 1)))
        _nn2_sample = tf.math.add(tf.math.reduce_sum(err_nn2, axis=0, keepdims=False), _vRel)
        indices = tf.constant(_relation_index)
        nn2 = tf.tensor_scatter_nd_update(nn2,tf.expand_dims([indices], 1), tf.reshape(_nn2_sample,(1, _nn2_sample.shape[0])))

        err_nn0 = tf.math.multiply( _nn1_samples, tf.reshape(g, (g.shape[0], 1)))
        _nn0_sample = tf.math.add(tf.math.reduce_sum(err_nn0, axis=0, keepdims=False),_vHead)
        indices = tf.constant(_head_index)
        nn0 = tf.tensor_scatter_nd_update(nn0,tf.expand_dims([indices], 1), tf.reshape(_nn0_sample,(1, _nn0_sample.shape[0])))

    def _save(self, step):        
        if self._save_embedings:
            np.save(os.path.join(self.path, "nn0.npy"), self.nn0.numpy())
            np.save(os.path.join(self.path, "nn1.npy"), self.nn1.numpy())
            np.save(os.path.join(self.path, "nn2.npy"), self.nn2.numpy())
            np.save(os.path.join(self.path, "x.npy"),step)
