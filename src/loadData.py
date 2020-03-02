import networkx as nx

def train(src , hIndex=0 , rIndex=1, tIndex=2):
    G = nx.DiGraph()
    _list = []
    file_edges = open(src, 'r')
    for line in file_edges:
        tokens = line.split()
        if G.has_node(tokens[hIndex]) is False:
            G.add_node(tokens[hIndex])
        if G.has_node(tokens[tIndex]) is False:
            G.add_node(tokens[tIndex])
        G.add_edge(tokens[hIndex], tokens[tIndex], label=tokens[rIndex], weight=0)
        _list.append(line.split())
    return G, _list

def test(src):
    f_test= open(src, 'r')
    test_list = []
    for line in f_test:
        test_list.append(line.split())
    f_test.close()
    return test_list       

# TODO : Optimiza needed
def read_relation(src):
    f_test = open(src, 'r')
    test_list = []
    count = []
    for line in f_test:
        line = line.split()
        relation = line[1].strip()
        if relation in test_list:
            index = test_list.index(relation)
            link = count[index]
            link +=1
            count[index]=link
        else:
            count.append(1)
            test_list.append(relation)
    f_test.close()
    l = []
    for index,r in enumerate(test_list):
        l.append((r,count[index]))
    return l

# TODO : Optimiza needed
def read_vocab(src):
    f_test = open(src, 'r')
    test_list = []
    count = []
    for line in f_test:
        line = line.split()
        head = line[0].strip()
        tail = line[2].strip()
        if head in test_list:
            index = test_list.index(head)
            link = count[index]
            link +=1
            count[index]=link
        else:
            count.append(1)
            test_list.append(head)
        if tail in test_list:
            index = test_list.index(tail)
            link = count[index]
            link +=1
            count[index]=link
        else:
            count.append(1)
            test_list.append(tail)  
    f_test.close()
    l = []
    for index,r in enumerate(test_list):
        l.append((r,count[index]))
    return l    