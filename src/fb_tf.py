
import tensorflow as tf
from Data import Data 
import l3vf2 as l3vf
import TableForNegativeSamples as t


def init_param():
    config = {}
    config['train_path'] = './data/WN18/train.txt'
    config['test_path'] = './data/WN18/test.txt'
    config['validation_path'] = './data/WN18/valid.txt'
    config['en1_idx'] = 0
    config['en2_idx'] = 1
    config['rel_idx'] = 2
    d = Data(config)
    param = {}
    param['data'] = d
    param['evaluation_metric'] = tf.constant('mean_rank', dtype=tf.string)
    param['iteration'] = tf.Variable(0)
    param['model'] = tf.constant(1)
    param['window'] = tf.constant(1)
    param['kns'] = tf.constant(25)
    param['alpha'] = tf.Variable(0.000001, dtype = tf.float16)
    param['beta'] = tf.Variable(0.0000001, dtype = tf.float16)
    param['dim'] = tf.constant(150)
    param['iter_count'] = tf.constant(5000)

    param['ns_size'] = tf.constant(1000)
    param['delta'] = tf.constant(20)
    param['k'] = tf.constant(0)
    z = tf.constant(0.5)
    param['nn1'] = tf.random.uniform( shape=(param['data'].sizeOfEntities, param['dim']),
      minval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32),'minval'), 
      maxval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32)),
      dtype=tf.dtypes.float32, seed=None, name='nn1')
    
    param['nn0'] = tf.random.uniform(shape=(param['data'].sizeOfEntities, param['dim']), 
      minval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32)), 
      maxval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32)),
      dtype=tf.dtypes.float32, seed=None, name='nn0')

    param['nn2'] = tf.random.uniform(shape=(param['data'].sizeOfRelations, param['dim']), 
      minval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32)), 
      maxval=tf.math.truediv(z , tf.cast(param['dim'],tf.float32)),
      dtype=tf.dtypes.float32, seed=None, name='nn1')


    param['table'] = t.TableForNegativeSamples(d.entities_size)
    return param    

def _main():
  param = init_param()
  l3vf.train_1(param)

if __name__ == "__main__":
    _main()