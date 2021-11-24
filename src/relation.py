'''
Created on Feb 15, 2017

@author: root
'''


class Word:
    def __init__(self, label , count):
        self.label = label
        self.count = count

class Relation:
    def __init__(self, rel):
        self.rel = []
        self.relt = []
        self.rel_map = {}
        self.total = 0
        self.maxx = 0
        self.build_rel(rel)
    
    def   build_rel(self, rel):
        relt =[]
        words = []
        word_map = {}
        tot = 0
        maxx = 0
        for token in rel:
            word_map[token[0]] = len(words)
            words.append(Word(token[0] , token[1]))
            tot +=token[1]
            if token[0] not in relt:
                relt.append(token[0])
            if token[1] > maxx:
                maxx = token[1]
        print ("\Relation built: %d" % len(words))
        self.rel = words
        self.rel_map = word_map  # Mapping from each token to its index in vocab
        self.relt = relt
        self.maxx = maxx
        self. total = tot
       

    def __getitem__(self, i):
        return self.rel[i]

    def __len__(self):
        return len(self.rel)

    def __iter__(self):
        return iter(self.rel)

    def __contains__(self, key):
        return key in self.rel_map

    def indices(self, tokens):
        return [self.rel_map[token] for token in tokens]

    def indice(self, token):
        return self.rel_map[token]
    
    def indices1(self, tokens):
        return [(self.rel_map[token[0]] , token[1]) for token in tokens]
    
    def getRel(self):
        return self.relt
