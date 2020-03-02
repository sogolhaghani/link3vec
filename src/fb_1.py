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

def timing(func):
    def wrap(*args):
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print( '%s function took %0.3f ms' %(func.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def init_param():
    param = {}
    train_list = lg.train('./data/freebase15k/train.txt')
    # param['graph'] = G
    param['train'] = train_list
    # param['validation'] = lg.test('./data/freebase15k/valid.txt')
    param['test'] = lg.test('./data/freebase15k/test.txt')
    param['relation'] = lg.read_relation('./data/freebase15k/train.txt')
    param['vocabulary'] = lg.read_vocab('./data/freebase15k/train.txt')

    param['evaluation_metric'] = 'mean_rank'
    param['iteration'] = 0
    param['model'] = 1
   

    param['window'] = 1
    param['kns'] = 20
    param['alpha'] = 0.07
    param['beta'] = 0.03
    param['dim'] = 200
    param['iter_count'] = 500

    param['ns_size'] = 1000
    param['delta'] = 20
    param['k'] = 0

    param['vocab'] = v.Vocabulary(param['vocabulary'])
    param['rel'] = r.Relation(param['relation'])
    param['relation'] = []
    param['vocabulary'] = []
    param['nn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(len(param['vocab']), param['dim']))
    param['nn1'] = np.zeros(shape=(len(param['vocab']), param['dim']))
    param['nn2'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(len(param['rel']), param['dim']))
    param['table'] = t.TableForNegativeSamples(param['vocab'])
    param['test_corrupted_all'] = Co._all(param, 'test')
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
