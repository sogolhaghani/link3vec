import numpy as np
from multiprocessing import Pool

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
    new_list = np.zeros(shape=(len(param[list_name]), 3), dtype=np.float32)
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
def create_filtered_corupted(head_index, rel_index, tail_index, vocab, train, test, validation):
    size = 0
    corupted = np.zeros(shape=(len(vocab), 4), dtype=np.float32)
    corupted2 = np.zeros(shape=(len(vocab), 4), dtype=np.float32)
    index = 0
    triple = [head_index, rel_index, tail_index]
    for word in  vocab:
        c_triple = [head_index, rel_index, vocab.indice(word.label)]
        if c_triple != triple and ( existt(c_triple, test) or  existt(c_triple, validation) or  existt(c_triple, train)):
            continue
        size += 1
        corupted[index][0] = head_index
        corupted[index][1] = rel_index
        corupted[index][2] = vocab.indice(word.label)
        index += 1
    # print('size %s' %size)
    index = 0
    size2 = 0
    for word in  vocab:
        c_triple = [vocab.indice(word.label), rel_index, tail_index]
        if c_triple != triple and ( existt(c_triple, test) or  existt(c_triple, validation) or  existt(c_triple, train)):
            continue
        size2 += 1
        corupted2[index][0] = vocab.indice(word.label)
        corupted2[index][1] = rel_index
        corupted2[index][2] = tail_index
        index += 1
    # print('size %s' %size2)
    return corupted[0: size], corupted2[0: size2], 

def create_corrupted(triple, _size):
    corupted = np.zeros(shape = (_size, 4))
    corupted2 = np.zeros(shape = (_size, 4))
    for index in range(_size):
        corupted[index][0] = triple[0]
        corupted[index][1] = triple[1]
        corupted[index][2] = index 
        corupted2[index][0] = index
        corupted2[index][1] = triple[1]
        corupted2[index][2] = triple[2]

    return corupted, corupted2

def _all(param):
    result = []
    for triple in param['data'].test:
        c_1, c_2 = create_corrupted(triple, param['data'].sizeOfEntities)
        result.append((c_1, c_2))
    return result        

def create_filtered_corupted_helper(args):
	return create_filtered_corupted(*args)

def _filtered(param, test_file):
    data = []
    vocab = param['vocab']
    rel = param['rel']
    train = convertToMatrixIndex(param, 'train')
    test = convertToMatrixIndex(param, 'test')
    validation = convertToMatrixIndex(param, 'validation')
    tasks = [] 

    for line_tokens in param[test_file]:
        head = line_tokens[0]
        relation = line_tokens[1]
        tail = line_tokens[2]
        head_index = vocab.indice(head)
        rel_index = rel.indice(relation)
        tail_index = vocab.indice(tail)
        tasks.append((head_index, rel_index, tail_index, vocab, train, test, validation))
    with Pool(4) as pool:  
        data = pool.map(create_filtered_corupted_helper,iterable=tasks)  
    return data


