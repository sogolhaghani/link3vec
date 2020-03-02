'''
Created on Feb 15, 2017
Modified on Aug 3, 2017
@author: root
'''
import tensorflow as tf
import math 
import numpy as np
import evaluate4 as e2
import activationFunction as af
import GenerateNewSample as gns
import FCSA as fcsa
from random import randint

from random import shuffle

def train_1(param):
    lv = param['iteration']
    tf.while_loop(lv < param['iter_count'] , link2vec3(param), loop_vars=lv,name='Main Loop')                        
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
        tail = walk[2]
        # neu1e = np.zeros(param['dim'])
        negative_samples = [(target, 0) for target in param['table'].sample(param['kns'])]
        classifiers = [(tail, 1)] + negative_samples
        for target, label in classifiers:
            if label == 0:
                relation_index = randint(0, param['data'].sizeOfRelations-1)
            z = np.dot(param['nn1'][target], (param['nn0'][head] + param['nn2'][relation_index]))
            p = af.sigmoid(z)
            g = param['alpha'] * (label - p)
            g1 = param['beta'] * (label - p)
            param['nn0'][head] = g * param['nn1'][target] + param['nn0'][head]  # Update nn1
            param['nn2'][relation_index] = g * param['nn1'][target] + param['nn2'][relation_index] 
            param['nn1'][target] += (g * param['nn0'][head] + g1 * param['nn2'][relation_index])  # Error to backpropagate to nn0
        # param['nn1'][tail] +=neu1e


def link2vec3(param):
    walks = param['data'].train
    tf.random.shuffle(walks, seed=None, name='Shuffle Train')
    for walk in walks:
        # walk = fcsa.selectNewTriple(param, walk)
        head = walk[0]
        relation_index = walk[1]
        tail = walk[2]
        negative_samples = [(target, 0) for target in param['table'].sample(param['kns'])]
        classifiers = [(tail, 1)] + negative_samples
        _type = tf.random.uniform([0, 1], minval=0, maxval=1, dtype=tf.int32)
        if _type == 0:
            for target, label in classifiers:
                if label == 0:
                    relation_index = randint(0, param['data'].sizeOfRelations-1)

                target_row = tf.gather(param['nn1'], target)
                head_row = tf.gather(param['nn0'], head)
                relation_row = tf.gather(param['nn2'], relation_index)

                z = tf.tensordot(target_row , (head_row + relation_row), axes=0)
                p = tf.math.sigmoid(z)
                g = param['alpha'] * (label - p)
                g1 = param['beta'] * (label - p)
                tf.scatter_update(param['nn0'], tf.constant(head), g * target_row + head_row )  # Update nn1
                
                param['nn2'][relation_index] = g * param['nn1'][target] + param['nn2'][relation_index] 
                param['nn1'][target] += (g * param['nn0'][head] + g1 * param['nn2'][relation_index])  # Error to backpropagate to nn0
        if _type == 1:
            neu1e = tf.zeros(param['dim'])
            for target, label in classifiers:
                if label == 0:
                    relation_index = randint(0, param['data'].sizeOfRelations-1)
                z = tf.tensordot(param['nn0'][tail], (param['nn1'][target] + param['nn2'][relation_index]),axes=0)
                p = af.sigmoid(z)
                g = param['alpha'] * (label - p)
                g1 = param['beta'] * (label - p)
                neu1e += (g * param['nn1'][target] + g1 * param['nn2'][relation_index])  # Error to backpropagate to nn0
                param['nn1'][target] += g * param['nn0'][tail]  # Update nn1
                param['nn2'][relation_index] += g * param['nn0'][tail]
            param['nn0'][tail] +=neu1e  
    
    # print('Iter: %s, alpha: %s, beta: %s, kn: %s, dim: %s' % (param['iteration'], param['alpha'], param['beta'], param['kns'], param['dim']))        
    # if  param['iteration']%10 == 0 and param['iteration'] > 200:    
    # e2.testAll(param)
    param['iteration'] = tf.reduce_sum(1, param['iteration']) 
    # return [tf.add(param['iteration'], 1)]   