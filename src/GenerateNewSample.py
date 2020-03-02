'''
Created on Feb 15, 2017
Modified on May 31, 2017
@author: root
'''

import numpy as np
import activationFunction as af
import random

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


def selectNewTriples(temp,param):
    if param['delta'] > len(temp):
        return temp
    maxx = param['rel'].maxx + 1
    t = []  
    for i in range(param['delta']):
        lc = []  
        for link in temp:
            lc.append(maxx-param['rel'].__getitem__(link[1]).count)
        start = 0 
        rand_index = random.randint(0, sum(lc) -1) 
        for index , link in enumerate(temp):
            if rand_index >= start and rand_index < start + lc[index]:
                t.append(temp[index])
                temp.remove(temp[index])
                break
            start += lc[index]
    return t

def generate_new_sample(param):
    size = param['ns_size']
    vocab = param['vocab']
    rel = param['rel']
    triples = convertToMatrixIndex(param, 'train')
    nt = param['train']
    print (' train size : %s' %len(nt))
    nodes = []
    nodes2 = []
    f= open("./data/triples.txt","w+")
    for i in range(size):
        nodes.append(selectNode(param))
    for i in range(15000):
        nodes2.append(param['vocab'].__getitem__(random.randint(0, len(param['vocab']) -1) ))
    for node1 in nodes:
        temp = []
        for relation in rel:
            for node2 in nodes2:
                node1_index = vocab.indice(node1.label)
                node2_index = vocab.indice(node2.label)
                relation_index = rel.indice(relation.label)
                t1 = [node1_index, relation_index, node2_index]
                t2 = [node2_index, relation_index, node1_index]
                p1 = af.sigmoid(np.dot( param['nn0'][node1_index, :], param['nn2'][relation_index, :] + param['nn1'][node2_index, :]))
                p2 = af.sigmoid(np.dot( param['nn0'][node2_index, :], param['nn2'][relation_index, :] + param['nn1'][node1_index, :]))
                if p1> 0.85 and  existt(t1, triples) is False:
                    temp.append(t1)
                if p2> 0.85 and  existt(t2, triples) is False:
                    temp.append(t2)                
        ntriples = selectNewTriples(temp,param)
        for t in ntriples:
            link = (vocab.__getitem__(t[0]).label , rel.__getitem__(t[1]).label,vocab.__getitem__(t[2]).label )
            nt.append(link)
            f.write(' '.join(str(s)+'\t' for s in link) + '\n')
    f.close()                
    print (' new train size : %s' %len(nt))
    param['train'] = nt
    return nt


def existt(triple , lis):
    a = np.searchsorted(lis[:,0], triple[0])
    if a == 0 and lis[0][0] != triple[0]:
        return False
    b = []
    found = False
    while(found is False):
        if len(lis[:,0]) > a and lis[a][0] == triple[0]:
            b.append(lis[a])
            a = a+1
        else:
            found = True            
        
    if any((triple == x).all() for x in b):
        return True
    return False

def convertToMatrixIndex(param, list_name):
    new_list = np.zeros(shape=(len(param[list_name]), 3), dtype=int)
    for  index, line_tokens in enumerate(param[list_name]):
        head = line_tokens[0]
        relation = line_tokens[1]
        tail = line_tokens[2]
        if  head not in param['vocab'] or tail not in param['vocab'] or relation not in param['rel']:
            continue
        head_index = param['vocab'].indice(head)
        rel_index = param['rel'].indice(relation)
        tail_index = param['vocab'].indice(tail)
        new_list[index][0] = head_index
        new_list[index][1] = rel_index
        new_list[index][2] = tail_index
    new_list = sorted(new_list, key=lambda a_entry: a_entry[0], reverse=False)
    return np.asarray(new_list)