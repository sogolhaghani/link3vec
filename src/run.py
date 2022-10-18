import os
import numpy as np
import tensorflow as tf

from model import Link3Vec
from evaluation import EvaluateModel


          
def _config_WN18():
    config = {}
    config['path'] = '../data/WN18_numpy/'
    config['train'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "train.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['validation'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "validation.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['test'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "test.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['relations'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "relations.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['entities'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "entities.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['_save_embedings'] = False
    config['read_Last_state'] = False
    config['iteration'] = tf.constant(20)
    config['dim'] = tf.constant(150)
    config['kns'] = tf.constant(1)
    config['kns_r'] = tf.constant(1)
    config['alpha'] = tf.Variable(0.02, dtype = tf.float64)
    config['beta'] = tf.Variable(0.00005, dtype = tf.float64)
    config['hW'] = tf.Variable(0.01, dtype = tf.float64)
    config['tW'] = tf.Variable(0.98, dtype = tf.float64)
    config['rW'] = tf.Variable(0.01, dtype = tf.float64)

    with tf.device('/cpu:0'):
        config['nSam_ent'] = tf.gather( tf.random.categorical([tf.gather(config['entities'], 2, axis=1)], tf.math.multiply(config['kns'],config['train'].shape[0]) , dtype=None, seed=None),0)
        config['nSam_rel'] = tf.gather( tf.random.categorical([tf.gather(config['relations'], 2, axis=1)], tf.math.multiply(config['kns_r'],config['train'].shape[0]) , dtype=None, seed=None),0) 
    return config  

def _config_WN18RR():
    config = {}
    config['path'] = '../data/WN18RR_numpy/'
    config['train'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "train.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['validation'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "validation.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['test'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "test.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['relations'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "relations.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['entities'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "entities.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['_save_embedings'] = False
    config['read_Last_state'] = False
    config['iteration'] = tf.constant(2000)
    config['dim'] = tf.constant(150)
    config['kns'] = tf.constant(15)
    config['kns_r'] = tf.constant(5)
    config['alpha'] = tf.Variable(0.07, dtype = tf.float64)
    config['beta'] = tf.Variable(0.007, dtype = tf.float64)

    with tf.device('/cpu:0'):
        config['nSam_ent'] = tf.gather( tf.random.categorical([tf.gather(config['entities'], 2, axis=1)], tf.math.multiply(config['kns'],config['train'].shape[0]) , dtype=None, seed=None),0)
        config['nSam_rel'] = tf.gather( tf.random.categorical([tf.gather(config['relations'], 2, axis=1)], tf.math.multiply(config['kns_r'],config['train'].shape[0]) , dtype=None, seed=None),0) 
    return config   

def _config_Freebase15k():
    config = {}
    config['path'] = '../data/freebase15k1_numpy/'
    config['train'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "train.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['validation'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "validation.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['test'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "test.npy") ).astype(dtype=np.int64), dtype=tf.int64)
    config['relations'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "relations.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['entities'] = tf.convert_to_tensor( np.load(os.path.join(config['path'], "entities.npy") ).astype(dtype=np.float64), dtype=tf.float64)
    config['_save_embedings'] = False
    config['read_Last_state'] = False
    config['iteration'] = tf.constant(2000)
    config['dim'] = tf.constant(200)
    config['kns'] = tf.constant(1)
    config['kns_r'] = tf.constant(1)
    config['alpha'] = tf.Variable(0.001, dtype = tf.float64)
    config['beta'] = tf.Variable(0.001, dtype = tf.float64)

    # with tf.device('/cpu:0'):
    #     config['nSam_ent'] = tf.gather( tf.random.categorical([tf.gather(config['entities'], 2, axis=1)], tf.math.multiply(config['kns'],config['train'].shape[0]) , dtype=None, seed=None),0)
    #     config['nSam_rel'] = tf.gather( tf.random.categorical([tf.gather(config['relations'], 2, axis=1)], tf.math.multiply(config['kns_r'],config['train'].shape[0]) , dtype=None, seed=None),0) 
    return config  

def _config_Freebase15k_237():
    config = {}
    config['path'] = '../data/freebase15k-237_numpy/'
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
    return config                




def train(config):
    model = Link3Vec(config=config)
    t = EvaluateModel(config=config, nn0=model.nn0, nn1=model.nn1, nn2=model.nn2)
    for x in range(model.startIndex , config['iteration']):
        train = tf.random.shuffle(config['train'], seed=None, name=None)
        tensor = []
        _tSize = 50000#train.shape[0]
        for tIndex in range(0 , _tSize):
            triple = tf.gather(train, tIndex)
            # tensor.append(model._score_and_update_head(triple, tIndex))  
            tensor.append((tf.math.multiply(model.tW, model._score_and_update_tail_2(triple, tIndex)) 
            + tf.math.multiply(model.rW,model._score_and_update_rel_2(triple, tIndex)) 
            + tf.math.multiply(model.hW,model._score_and_update_head_2(triple, tIndex)) ) )   
        if x > 10:     
            print(x,sum(tensor) / _tSize)
        if tf.math.reduce_all(tf.math.is_nan(model.nn0)) or tf.math.reduce_all(tf.math.is_nan(model.nn1)) or tf.math.reduce_all(tf.math.is_nan(model.nn2)):
            print("fill with nan") 
            break
        model._save(x)
        if x > 2000:
            t.updateParameter(nn0=model.nn0, nn1=model.nn1, nn2=model.nn2)     
            t.eval()

              
if __name__ == "__main__":
    step = tf.Variable(0.02, dtype = tf.float64)
    config = _config_WN18()
    config['hW'] = tf.Variable(0, dtype = tf.float64)
    config['tW'] = tf.Variable(1, dtype = tf.float64)
    config['rW'] = tf.Variable(0, dtype = tf.float64)
    print(config['hW'], config['tW'], config['rW'])
    train(config)
    config['hW'] = tf.Variable(1, dtype = tf.float64)
    config['tW'] = tf.Variable(0, dtype = tf.float64)
    config['rW'] = tf.Variable(0, dtype = tf.float64)   
    print(config['hW'], config['tW'], config['rW'])
    train(config)
    config['hW'] = tf.Variable(0, dtype = tf.float64)
    config['tW'] = tf.Variable(0, dtype = tf.float64)
    config['rW'] = tf.Variable(1, dtype = tf.float64)  
    print(config['hW'], config['tW'], config['rW']) 
    train(config)
    config['hW'] = tf.Variable(0.33, dtype = tf.float64)
    config['tW'] = tf.Variable(0.34, dtype = tf.float64)
    config['rW'] = tf.Variable(0.33, dtype = tf.float64)  
    print(config['hW'], config['tW'], config['rW']) 
    train(config)
    config['hW'] = tf.Variable(0.15, dtype = tf.float64)
    config['tW'] = tf.Variable(0.34, dtype = tf.float64)
    config['rW'] = tf.Variable(0.51, dtype = tf.float64) 
    print(config['hW'], config['tW'], config['rW'])  
    train(config)
    config['hW'] = tf.Variable(0.51, dtype = tf.float64)
    config['tW'] = tf.Variable(0.34, dtype = tf.float64)
    config['rW'] = tf.Variable(0.15, dtype = tf.float64) 
    print(config['hW'], config['tW'], config['rW'])  
    train(config)
    config['hW'] = tf.Variable(0.34, dtype = tf.float64)
    config['tW'] = tf.Variable(0.51, dtype = tf.float64)
    config['rW'] = tf.Variable(0.15, dtype = tf.float64)  
    print(config['hW'], config['tW'], config['rW']) 
    train(config)
    config['hW'] = tf.Variable(0.34, dtype = tf.float64)
    config['tW'] = tf.Variable(0.15, dtype = tf.float64)
    config['rW'] = tf.Variable(0.51, dtype = tf.float64)
    print(config['hW'], config['tW'], config['rW'])   
    train(config)
    config['hW'] = tf.Variable(0.51, dtype = tf.float64)
    config['tW'] = tf.Variable(0.15, dtype = tf.float64)
    config['rW'] = tf.Variable(0.34, dtype = tf.float64)  
    print(config['hW'], config['tW'], config['rW']) 
    train(config)
    config['hW'] = tf.Variable(0.15, dtype = tf.float64)
    config['tW'] = tf.Variable(0.51, dtype = tf.float64)
    config['rW'] = tf.Variable(0.34, dtype = tf.float64) 
    print(config['hW'], config['tW'], config['rW'])  
    train(config)