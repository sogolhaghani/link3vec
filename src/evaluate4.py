'''
Created on Feb 1, 2017
Modified on April 21, 2017
Modified on June 2, 2017
@author: root
'''

import numpy as np
import concurrent.futures
import activationFunction as af
import Corrupted as Co


def testAll(param):
    _name = 'test'
    _type = '_corrupted_all'
    return test(param['data'].test, _type, param)

def test(testSet, _type, param):
    hits_1 = np.zeros(shape=(len(testSet), 1))
    hits_10 = np.zeros(shape=(len(testSet), 1))
    hits_3 = np.zeros(shape=(len(testSet), 1))
    ranks = np.zeros(shape=(len(testSet), 1))
    for  index in range(len(testSet)):
        rank, hit_1, hit_3, hit_10 = calculate(param, index, _type, testSet)
        hits_1[index] = hit_1
        hits_3[index] = hit_3
        hits_10[index] = hit_10
        ranks[index] = rank
    ranks = sorted(ranks, key=lambda a_entry: a_entry[0], reverse=False)
    avg_rank = np.mean(ranks)
    mean_rank = ranks[int(len(ranks)/2)]
    hits_1 = np.average(hits_1)
    hits_3 = np.average(hits_3)
    hits_10 = np.average(hits_10)
    MRR = mrr(ranks)
    # print('Avg Rank: %s' %avg_rank)
    # print( 'Mean Rank: %s' %mean_rank)
    # print( 'Hits %s : %s' %(10, hits_10))
    # print( 'Hits %s : %s' %(1, hits_1))
    # print( 'Hits %s : %s' %(3, hits_3))
    # print( 'MRR : %s ' %MRR)
    return MRR

def calculate(param , index, _type, testSet):
    top_hits_10 = 10
    top_hits_1 = 1
    top_hits_3 = 3
    triple = testSet[index]
    corupted, corupted2 = Co.create_corrupted(triple, param['data'].sizeOfEntities)   
    __calculate_scores_when_rel_is_vector(corupted, corupted2, param)
    corupted = sorted(corupted, key=lambda a_entry: a_entry[3], reverse=True)
    corupted2 = sorted(corupted2, key=lambda a_entry: a_entry[3], reverse=True)
    corupted = np.asarray(corupted)
    corupted2 = np.asarray(corupted2)
    rank = __mean_rank(corupted, corupted2, triple)
    hit_1 = __hits_top(corupted, corupted2, triple, top_hits_1)
    hit_10 = __hits_top(corupted, corupted2,triple, top_hits_10)
    hit_3 = __hits_top(corupted, corupted2, triple, top_hits_3)
    return rank, hit_1, hit_3, hit_10


def mrr(ranks):
    inverse = []
    for rank in ranks:
        inverse.append(float(1)/rank)
    summ = sum(inverse)
    return float(1)/len(inverse) * summ

def __mean_rank(corupted, corupted2, triple):
    rank = 4000
    rank2 = 4000
    for index in range(4000):
        c_head_index = corupted[index][0]
        c_rel_index = corupted[index][1]
        c_tail_index = corupted[index][2]
        if c_head_index == triple[0] and c_rel_index == triple[1] and c_tail_index == triple[2]:
            rank = index+1
            break
    for index in range(4000):
        c_head_index = corupted2[index][0]
        c_rel_index = corupted2[index][1]
        c_tail_index = corupted2[index][2]
        if c_head_index == triple[0] and c_rel_index == triple[1] and c_tail_index == triple[2]:
            rank2 = index+1
            break
    if rank < rank2:
        return rank
    return rank2            

def __hits_top(corupted, corupted2,triple, top_hits):
    for index in range(top_hits):
        c_head_index = corupted[index][0]
        c_rel_index = corupted[index][1]
        c_tail_index = corupted[index][2]
        if c_head_index == triple[0] and c_rel_index == triple[1] and c_tail_index == triple[2]:
            return 1
    for index in range(top_hits):
        c_head_index = corupted2[index][0]
        c_rel_index = corupted2[index][1]
        c_tail_index = corupted2[index][2]
        if c_head_index == triple[0] and c_rel_index == triple[1] and c_tail_index == triple[2]:
            return 1
    return 0

def __calculate_scores_when_rel_is_vector(corupted, corupted2, param):
    for index in range(len(corupted[:, 0])):
        head_index = corupted[index][0]
        rel_index = corupted[index][1]
        tail_index = corupted[index][2]
        v_h = param['nn0'][head_index.astype(int), :]
        v_r = param['nn2'][rel_index.astype(int) , :]
        v_t = param['nn1'][tail_index.astype(int), :]
        z_param = np.dot(v_h, v_t + v_r)
        corupted[index][3] = af.sigmoid(z_param)
    for index in range(len(corupted2[:, 0])):
        head_index = corupted2[index][0]
        rel_index = corupted2[index][1]
        tail_index = corupted2[index][2]
        v_h = param['nn0'][head_index.astype(int), :]
        v_r = param['nn2'][rel_index.astype(int) , :]
        v_t = param['nn1'][tail_index.astype(int), :]
        z_param = np.dot(v_h, v_t + v_r)
        corupted2[index][3] = af.sigmoid(z_param)
    return corupted, corupted2
