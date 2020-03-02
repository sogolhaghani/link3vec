

'''
Created on Feb 15, 2017
Modified on May 31, 2017
@author: root
'''
import math
import random
import numpy as np
import activationFunction as af

def selectNewTriple(param, walk):
    if param['k'] == 0 :
        return walk
    
    z = param['iter_count'] - param['iteration']
    epsilon = min(float(param['k']) / (param['k'] + math.exp(z)) , 0.15)
    if np.random.uniform() > epsilon:
        return walk
    nodes2 = []
    vocab = param['vocab']
    rel = param['rel']
    for i in range(5000):
        nodes2.append(vocab.__getitem__(random.randint(0, len(vocab) -1) ))
        # nodes2.append(selectNode(param))
    for relation in rel:
        temp = []
        for node2 in nodes2:
            node1_index = vocab.indice(walk[0])
            node2_index = vocab.indice(node2.label)
            relation_index = rel.indice(relation.label)
            t1 = [node1_index, relation_index, node2_index]
            p1 = af.sigmoid(np.dot( param['nn0'][node1_index, :], param['nn2'][relation_index, :] + param['nn1'][node2_index, :]))
            if p1> 0.75:
                temp.append(t1)
    if len(temp) == 0:
        return walk
    triple_index = selectTriples(temp,param)
    return (vocab.__getitem__(triple_index[0]).label , rel.__getitem__(triple_index[1]).label,vocab.__getitem__(triple_index[2]).label )

def selectNode(param):
    maxx = param['vocab'].maxx + 1
    lc = []  
    for node in param['vocab']:
        lc.append(maxx-node.count)
    rand_index = random.randint(0, param['vocab'].total -1) 
    start = 0 
    for index , node in enumerate(param['vocab']):
        if rand_index <= start and rand_index > start + lc[index]:
            return node
        start += lc[index]
    return param['vocab'].__getitem__(random.randint(0, len(param['vocab']) -1) )


def selectTriples(temp,param):
    maxx = param['rel'].maxx + 1
    lc = []  
    for link in temp:
        lc.append(maxx-param['rel'].__getitem__(link[1]).count)
    rand_index = random.randint(0, sum(lc) -1) 
    start = 0 
    for index , link in enumerate(temp):
        if rand_index >= start and rand_index < start + lc[index]:
            return link
        start += lc[index]
    return temp[random.randint(0, len(temp) -1)]