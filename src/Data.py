import tensorflow as tf
from os import path

# TODO : Refactor. load from saved data
class Data:

    def __init__(self, config):
        self.train = []
        self.test = []
        self.validation = []
        self.entities_size = []
        self.relations_size = []
        self.sizeOfEntities = tf.Variable(0, dtype=tf.int32)
        self.sizeOfRelations = tf.Variable(0)
        self.entities_name = []
        self.relations_name = []

        self.read_files(config)

    # def makeCheckpoint(self):
    #     return tf.train.Checkpoint(
    #         train=self.train, test=self.test, validation=self.validation,
    #         entities_size=self.entities_size, relations_size=self.relations_size, entities_name=self.entities_name,
    #         relations_name = self.relations_name,sizeOfEntities =self.sizeOfEntities, sizeOfRelations = self.sizeOfRelations)

    # def restoreVariables(self):
    #     status = self.ckpt.restore(tf.train.latest_checkpoint('.'))
    #     status.assert_consumed()  # Optional check

    def read_files(self, config):
        # if path.exists("./ckpt"):
        #     self.restoreVariables()
        #     return

        _vocab_list = []
        _vocab_list_size = []
        _rel_list = []
        _rel_list_size = []
        triples = []
        file_edges = open(config['train_path'], 'r')
        for line in file_edges:
            tokens = line.split()
            i_e_1 = 0
            i_e_2 = 0
            i_r = 0
            if tokens[0] in _vocab_list:
                i_e_1 = _vocab_list.index(tokens[0])
                _vocab_list_size[i_e_1] += 1 
            else:    
                _vocab_list.append(tokens[0])
                _vocab_list_size.append(1)
                i_e_1 = len(_vocab_list) - 1
            
            if tokens[2] in _vocab_list:
                i_e_2 = _vocab_list.index(tokens[2])
                _vocab_list_size[i_e_2] += 1 
            else:    
                _vocab_list.append(tokens[2])
                _vocab_list_size.append(1)
                i_e_2 = len(_vocab_list) - 1

            
            if tokens[1] in _rel_list:
                i_r = _rel_list.index(tokens[1])
                _rel_list_size[i_r] += 1 
            else:    
                _rel_list.append(tokens[1])
                _rel_list_size.append(1)
                i_r = len(_rel_list) - 1
            triples.append((i_e_1 , i_r , i_e_2))
        file_edges.close()
        self.entities_name = tf.convert_to_tensor(_vocab_list, dtype=tf.string) 
        self.entities_size = tf.convert_to_tensor(_vocab_list_size, dtype=tf.uint16)  
        self.relations_name = tf.convert_to_tensor(_rel_list, dtype=tf.string) 
        self.relations_size = tf.convert_to_tensor(_rel_list_size, dtype=tf.uint16)  
        self.sizeOfEntities = tf.constant(len(_vocab_list_size))
        self.sizeOfRelations =tf.constant(len(_rel_list_size))
        self.train = tf.convert_to_tensor(triples, dtype=tf.uint16)
        _test_triples = []
        file_edges = open(config['test_path'], 'r')
        for line in file_edges:
            tokens = line.split()
            i_e_1 = -1
            i_e_2 = -1
            i_r = -1
            if tokens[0] in _vocab_list:
                i_e_1 = _vocab_list.index(tokens[0])
            else:    
               print('%s not in entities' ,tokens[0])
            
            if tokens[2] in _vocab_list:
                i_e_2 = _vocab_list.index(tokens[2])
            else:    
                print('%s not in entities' ,tokens[2])           
            if tokens[1] in _rel_list:
                i_r = _rel_list.index(tokens[1])
            else:    
                print('%s not in relation' ,tokens[1])  
            _test_triples.append((i_e_1 , i_r , i_e_2))
        self.test = tf.convert_to_tensor(_test_triples, dtype=tf.uint16)

        _validations_triples = []
        file_edges = open(config['validation_path'], 'r')
        for line in file_edges:
            tokens = line.split()
            i_e_1 = -1
            i_e_2 = -1
            i_r = -1
            if tokens[0] in _vocab_list:
                i_e_1 = _vocab_list.index(tokens[0])
            else:    
               print('%s not in entities' ,tokens[0])
            
            if tokens[2] in _vocab_list:
                i_e_2 = _vocab_list.index(tokens[2])
            else:    
                print('%s not in entities' ,tokens[2])           
            if tokens[1] in _rel_list:
                i_r = _rel_list.index(tokens[1])
            else:    
                print('%s not in relation' ,tokens[1])  
            _test_triples.append((i_e_1 , i_r , i_e_2))
        self.validation = tf.convert_to_tensor(_validations_triples, dtype=tf.uint16)            
        # self.ckpt.save('./ckpt')