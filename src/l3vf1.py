'''
Created on Feb 15, 2017
Modified on Aug 3, 2017
@author: root
'''
import math 
import numpy as np
import evaluate4 as e2
import activationFunction as af
import walk as w
import GenerateNewSample as gns
import FCSA as fcsa
from random import randint

from random import shuffle

def train_1(param):
    while param['iteration'] < param['iter_count']:                              
        link2vec1(param)
        print('Iter: %s, alpha: %s, beta: %s, kn: %s, dim: %s' % (param['iteration'], param['alpha'], param['beta'], param['kns'], param['dim']))        
        if  param['iteration']%10 == 0 and param['iteration'] > 200:    
            e2.testAll(param)
        param['iteration'] += 1
    return param


def link2vec1(param):
    walks = param['data'].train
    shuffle(walks)
    for walk in walks:
        walk = fcsa.selectNewTriple(param, walk)
        head = walk[0]
        relation_index = walk[1]
        contexts = [walk[2]]
        for tail in contexts:
            neu1e = np.zeros(param['dim'])
            negative_samples = [(target, 0) for target in param['table'].sample(param['kns'])]
            classifiers = [(head, 1)] + negative_samples
            for target, label in classifiers:
                if label == 0:
                    relation_index = randint(0, param['data'].sizeOfRelations-1)
                z = np.dot(param['nn0'][tail], (param['nn1'][target] + param['nn2'][relation_index]))
                p = af.sigmoid(z)
                g = param['alpha'] * (label - p)
                g1 = param['beta'] * (label - p)
                neu1e += (g * param['nn1'][target] + g1 * param['nn2'][relation_index])  # Error to backpropagate to nn0
                param['nn1'][target] += g * param['nn0'][tail]  # Update nn1
                param['nn2'][relation_index] += g * param['nn0'][tail]
            param['nn0'][tail] +=neu1e  


def link2vec2(param):
    walks = param['data'].train
    shuffle(walks)
    for walk in walks:
        walk = fcsa.selectNewTriple(param, walk)
        head = walk[0]
        relation_index = walk[1]
        tail = [walk[2]]    
        neu1e = np.zeros(param['dim'])
        negative_samples = [(target, 0) for target in param['table'].sample(param['kns'])]
        classifiers = [(tail, 1)] + negative_samples
        for target, label in classifiers:
            if label == 0:
                relation_index = randint(0, param['data'].sizeOfRelations-1)
            z = np.dot(param['nn1'][target], (param['nn0'][head] + param['nn2'][relation_index]))
            p = af.sigmoid(z)
            g = param['alpha'] * (label - p)
            g1 = param['beta'] * (label - p)
            param['nn1'][tail] += (g * param['nn0'][head] + g1 * param['nn2'][relation_index])  # Error to backpropagate to nn0
            param['nn0'][head] = g * param['nn1'][target] + param['nn0'][head]  # Update nn1
            param['nn2'][relation_index] = g * param['nn1'][target] + param['nn2'][relation_index] 
        # param['nn1'][tail] +=neu1e