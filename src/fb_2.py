'''
Created on Jan 28, 2017
Modified on May 31, 2017

@author: Sogol

wn18 final model
'''
import time
import numpy as np
import loadData as lg
import l3vf1 as l2v
import chart as c
import evaluate4 as e
import TableForNegativeSamples as t
import vocabulary as v
import relation as r
import Corrupted as Co
import Data as da

def timing(func):
    def wrap(*args):
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print( '%s function took %0.3f ms' %(func.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def init_param():

    config = {}
    config['train_path'] = './data/freebase15k/train.txt'
    config['test_path'] = './data/freebase15k/train-1.txt'
    config['validation_path'] = './data/freebase15k/valid.txt'
    d = da.Data(config)
    param = {}
    param['data'] = d
    param['evaluation_metric'] = 'mean_rank'
    param['iteration'] = 0
    param['model'] = 1
    param['window'] = 1
    param['kns'] = 1
    param['alpha'] = 0.001
    param['beta'] = 0.0001
    param['dim'] = 150
    param['iter_count'] = 5000

    param['ns_size'] = 1000
    param['delta'] = 20
    param['k'] = 0
    param['nn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfEntities, param['dim']))
    param['nn1'] = np.zeros(shape=(param['data'].sizeOfEntities, param['dim']))
    param['nn2'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfRelations, param['dim']))
    param['table'] = t.TableForNegativeSamples(d.entities_size)
    # param['test_corrupted_all'] = Co._all(param)
    # param['test_corrupted_filtered'] = Co._filtered(param, 'test')
    # param['validation_corrupted'] = Co._all(param, 'validation')
    # param['validation_corrupted_filtered'] = Co._filtered(param, 'validation')
    return param    

@timing
def _main():
    param = init_param()
    l2v.train_1(param)
    # print 'test with raw setting'
    # e2.test(param, 'test', False, False)
    # print 'test with filter setting'
    # e2.test(param, 'test', False, True)

if __name__ == "__main__":
    _main()
