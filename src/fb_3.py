import time
import pickle
import numpy as np
import l3vf2 as l2v
import TableForNegativeSamples as t

def timing(func):
    def wrap(*args):
        time1 = time.time()
        ret = func(*args)
        time2 = time.time()
        print( '%s function took %0.3f ms' %(func.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def init_param():
    with open('./data/freebase15k/data.pkl', 'rb') as input:
        d = pickle.load(input)
    param = {}
    param['data'] = d
    param['evaluation_metric'] = 'mean_rank'
    param['iteration'] = 0
    param['model'] = 1
    param['window'] = 1
    param['kns'] = 25
    param['alpha'] = 0.000001
    param['beta'] = 0.0000001
    param['dim'] = 150
    param['iter_count'] = 5000

    param['ns_size'] = 1000
    param['delta'] = 20
    param['k'] = 0
    param['nn1'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfEntities, param['dim']))
    param['nn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfEntities, param['dim']))
    param['nn2'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfRelations, param['dim']))
    param['table'] = t.TableForNegativeSamples(d.entities_size)
    return param    

@timing
def _main():
    param = init_param()
    l2v.train_1(param)

if __name__ == "__main__":
    _main()
