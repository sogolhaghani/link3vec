import os
import numpy as np
import tensorflow as tf
from scipy.stats import rankdata

class EvaluateModel():
    def __init__(self,config, nn0,nn1,nn2):
        self.nn0 = nn0
        self.nn1 = nn1
        self.nn2 = nn2
        self.entities = config['entities']
        self.relations = config['relations']
        self.train = config['train']
        self.test = config['test']
        self.validation = config['validation']
        self.hW = config['hW']
        self.rW = config['rW']
        self.tW = config['tW']

    def updateParameter(self, nn0,nn1,nn2):
        self.nn0 = nn0
        self.nn1 = nn1
        self.nn2 = nn2    

    def calculateRank(self, triple):  
        selectedEntities = tf.cast(tf.gather(self.entities, 0, axis=1), dtype=tf.int64)
        _head_index = tf.gather(triple,0)
        _tail_index = tf.gather(triple,1)
        _relation_index = tf.gather(triple,2)  

        indexes = tf.where( tf.math.logical_and( (tf.gather(self.train, 1, axis=1) == _tail_index) , (tf.gather(self.train, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.train, indexes), 0, axis=1)
        shape = tf.constant([self.entities.shape[0]], dtype=tf.int64)   
        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesTrain = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 

        indexes = tf.where( tf.math.logical_and( (tf.gather(self.test, 1, axis=1) == _tail_index) , (tf.gather(self.test, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.test, indexes), 0, axis=1)
        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesTest = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 
        indexes = tf.where( tf.math.logical_and( (tf.gather(self.validation, 1, axis=1) == _tail_index) , (tf.gather(self.validation, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.validation, indexes), 0, axis=1)
        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesvalidation = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 

        selectedEntitiesTrain= tf.reshape(selectedEntitiesTrain, (1, selectedEntitiesTrain.shape[0]))
        selectedEntitiesTest = tf.reshape(selectedEntitiesTest, (1, selectedEntitiesTest.shape[0]))
        selectedEntitiesvalidation = tf.reshape(selectedEntitiesvalidation, (1, selectedEntitiesvalidation.shape[0]))
        selectedEntitiesFinal = tf.sets.intersection(selectedEntitiesTrain, selectedEntitiesTest)
        selectedEntitiesFinal = tf.cast(tf.sets.intersection(selectedEntitiesFinal, selectedEntitiesvalidation), dtype=tf.int64).values
        couraptedH = tf.concat([  tf.reshape(selectedEntitiesFinal, (selectedEntitiesFinal.shape[0], 1)) , tf.fill([selectedEntitiesFinal.shape[0], 1], _tail_index) , tf.fill([selectedEntitiesFinal.shape[0], 1], _relation_index)],1)
        couraptedH = tf.concat([couraptedH, tf.reshape(triple, (1,3))], 0)
        indexes = None
        existEntites = None
        selectedEntitiesTrain = None
        selectedEntitiesTest = None
        selectedEntitiesvalidation = None
        selectedEntitiesFinal = None
        div = tf.constant(3)
        sliceSize = tf.cast(tf.math.floor(tf.math.truediv(couraptedH.shape[0], div)),dtype=tf.int64)
        s0, s1, s2= tf.split(couraptedH, num_or_size_splits=[
            sliceSize , 
            sliceSize , 
            tf.math.subtract(couraptedH.shape[0] , tf.math.multiply(sliceSize, 2)) ], axis=0)
        couraptedH = None

        p0 =  tf.linalg.diag_part(self.calcProbability(s0,_relation_index))
        p1 =  tf.linalg.diag_part(self.calcProbability(s1,_relation_index))
        p2 =  tf.linalg.diag_part(self.calcProbability(s2,_relation_index))

        p = tf.math.subtract(1 , tf.concat([p0,p1,p2,], 0))
        ranks = rankdata(p.numpy(), method='min')
        rankH = ranks[-1]
        pH = p.numpy()[-1]
        p0 = None
        p1 = None
        p2 = None
        p = None

        indexes = tf.where( tf.math.logical_and( (tf.gather(self.train, 0, axis=1) == _head_index) , (tf.gather(self.train, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.train, indexes), 1, axis=1)
        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesTrain = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 
        indexes = tf.where( tf.math.logical_and( (tf.gather(self.test, 0, axis=1) == _head_index) , (tf.gather(self.test, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.test, indexes), 1, axis=1)

        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesTest = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 
        indexes = tf.where( tf.math.logical_and( (tf.gather(self.validation, 0, axis=1) == _head_index) , (tf.gather(self.validation, 2, axis=1) == _relation_index)))
        indexes = tf.reshape(indexes, (indexes.shape[0] ))
        existEntites = tf.gather(tf.gather(self.validation, indexes), 1, axis=1)

        scatter = tf.scatter_nd(tf.reshape(existEntites ,  (existEntites.shape[0] , 1) ), tf.math.add(existEntites ,1), shape)
        selectedEntitiesvalidation = tf.where(tf.math.greater(tf.subtract(selectedEntities , scatter), -1)) 
        selectedEntitiesTrain= tf.reshape(selectedEntitiesTrain, (1, selectedEntitiesTrain.shape[0]))
        selectedEntitiesTest = tf.reshape(selectedEntitiesTest, (1, selectedEntitiesTest.shape[0]))
        selectedEntitiesvalidation = tf.reshape(selectedEntitiesvalidation, (1, selectedEntitiesvalidation.shape[0]))

        selectedEntitiesFinal = tf.sets.intersection(selectedEntitiesTrain, selectedEntitiesTest)
        selectedEntitiesFinal = tf.cast(tf.sets.intersection(selectedEntitiesFinal, selectedEntitiesvalidation), dtype=tf.int64).values
        couraptedT = tf.concat([   tf.fill([selectedEntitiesFinal.shape[0], 1], _head_index) , tf.reshape(selectedEntitiesFinal, (selectedEntitiesFinal.shape[0], 1)) , tf.fill([selectedEntitiesFinal.shape[0], 1], _relation_index)],1)
        couraptedT = tf.concat([couraptedT, tf.reshape(triple, (1,3))], 0)    
        indexes = None
        existEntites = None
        selectedEntitiesTrain = None
        selectedEntitiesTest = None
        selectedEntitiesvalidation = None
        selectedEntitiesFinal = None
        div = tf.constant(3)
        sliceSize = tf.cast(tf.math.floor(tf.math.truediv(couraptedT.shape[0], div)),dtype=tf.int64)
        s0, s1, s2= tf.split(couraptedT, num_or_size_splits=[
            sliceSize , 
            sliceSize , 
            tf.math.subtract(couraptedT.shape[0] , tf.math.multiply(sliceSize, 2)) ], axis=0)
        couraptedT = None

        p0 =  tf.linalg.diag_part(self.calcProbability(s0,_relation_index))
        p1 =  tf.linalg.diag_part(self.calcProbability(s1,_relation_index))
        p2 =  tf.linalg.diag_part(self.calcProbability(s2,_relation_index))

        p = tf.math.subtract(1 , tf.concat([p0,p1,p2,], 0))
        ranks = rankdata(p.numpy(), method='min')    
        rankT = ranks[-1]
        pT = p.numpy()[-1]
        p = None
        if tf.math.greater(rankH , rankT):
            return rankT , pT
        else:
            return rankH , pH  

    def calcProbability(self,s, _relation_index):               
        return tf.math.multiply(self.hW,tf.math.sigmoid(tf.tensordot(  tf.gather(self.nn0,tf.gather(s,0, axis=1),0),tf.transpose(  tf.gather( self.nn1,  tf.gather( s,1, axis=1)) + tf.gather(self.nn2,_relation_index) ),axes=1 )) ) + tf.math.multiply(self.tW,tf.math.sigmoid(tf.tensordot(  tf.gather(self.nn1,tf.gather(s,1, axis=1),0),tf.transpose(  tf.gather( self.nn0,  tf.gather( s,0, axis=1)) + tf.gather(self.nn2,_relation_index) ),axes=1 )) ) + tf.math.multiply(self.rW,tf.math.sigmoid(tf.tensordot(  tf.gather(self.nn2,_relation_index),tf.transpose(  tf.gather( self.nn1,  tf.gather( s,1, axis=1)) + tf.gather(self.nn0,tf.gather(s,0, axis=1),0) ),axes=1 )) )
        
        # return tf.math.sigmoid(tf.tensordot(  tf.gather(self.nn1,tf.gather(s,1, axis=1),0),tf.transpose(  tf.gather( self.nn0,  tf.gather( s,0, axis=1)) + tf.gather(self.nn2,_relation_index) ),axes=1 )) 

    def mrr(self, ranks):
        inverse = []
        one = tf.constant(1.0, dtype = tf.float64)
        for rank in ranks:
            inverse.append( tf.math.truediv(one , tf.cast(rank,tf.float64)))
        summ = tf.reduce_sum(inverse)
        return tf.math.multiply(tf.math.truediv(one , len(inverse)) , summ)

    def HitAtK(self, ranks, k=10):
        _hitAt = 0
        for rank in ranks:
            if rank <= k:
                _hitAt+=1
        return tf.math.truediv(_hitAt , len(ranks))        

    def mr(self, ranks):
        return tf.math.truediv(tf.reduce_sum(ranks) , len(ranks))     

    def eval(self, evalSet='validation'):
        tensor = []
        probs = []
        _set = self.validation
        if evalSet == 'test':
            _set = self.test
        # _set = tf.random.shuffle(_set, seed=None, name=None)            
        for tIndex in range(0 ,100):
        # for tIndex in range(0 ,_set.shape[0]):            
            triple = tf.gather(_set,tIndex)
            r, p = self.calculateRank(triple)
            tensor.append(r)
            probs.append(p)
        print("MRR : %10f" %self.mrr(tensor))
        print("MR : %10f" %self.mr(tensor))
        print("Hit @ 10 : %10f" %self.HitAtK(tensor))
        print("Hit @ 3 : %10f" %self.HitAtK(tensor, k=3))
        print("Hit @ 1 : %10f" %self.HitAtK(tensor, k=1))
        # print(probs)
        # print(tensor)
