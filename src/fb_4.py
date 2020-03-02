import time
import pickle
import numpy as np
import l3vf2 as l2v
import TableForNegativeSamples as t
import evaluate4 as e2


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
    param['iter_count'] = 100

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
    _alpha = 0.00000001
    _alpha_step = 0.00000005
    _beta_step = 0.00000005
    _k_step = 3

    best_alpha = 0
    best_beta = 0
    best_k = 0
    best_result = 0
    best_iter = 0

    param = init_param()
    while _alpha < 1:    
        _beta = 0.00000001
        while _beta < 1:    
            _k = 1
            while _k < 50:
                param['kns'] = _k
                while param['iteration'] < param['iter_count']:                              
                    l2v.link2vec3(param)
                    # print('Iter: %s, alpha: %s, beta: %s, kn: %s, dim: %s' % (param['iteration'], param['alpha'], param['beta'], param['kns'], param['dim']))        
                    # if  param['iteration']%10 == 0 and param['iteration'] > 200:    
                    mrr = e2.testAll(param)
                    if mrr > best_result:
                        best_alpha = param['alpha']
                        best_beta = param['beta']
                        best_k = param['kns']
                        best_result = mrr
                        best_iter = param['iteration']
                    param['iteration'] += 1
        
                param['iteration'] = 0
                param['nn1'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfEntities, param['dim']))
                param['nn0'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfEntities, param['dim']))
                param['nn2'] = np.random.uniform(low=-0.5 / param['dim'], high=0.5 / param['dim'], size=(param['data'].sizeOfRelations, param['dim']))

                _k = _k + _k_step
                param['kns'] = _k

            _beta = _beta + _beta_step
            param['beta'] = _beta
        
        _alpha = _alpha + _alpha_step
        param['alpha'] = _alpha
    print(best_alpha)
    print(best_beta)
    print(best_k)
    print(best_result)
    print(best_iter)
                
    return param

if __name__ == "__main__":
    _main()
