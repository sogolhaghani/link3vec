'''
Created on Feb 15, 2017

@author: root
'''


class Word:
    def __init__(self, label , count):
        self.label = label
        self.count = count

class Vocabulary:
    def __init__(self, vocab):
        self.words = []
        self.word_map = {}
        self.total = 0
        self.maxx = 0
        self.build_words(vocab)

#     @timing
    def build_words(self, vocab):
        words = []
        word_map = {}
        tot = 0
        maxx = 0
        print ('vocab len : %s'  %len(vocab))
        for token in vocab:
            word_map[token[0]] = len(words)
            words.append(Word(token[0] , token[1]))
            tot +=int(token[1])
            if token[1] > maxx:
                maxx = token[1]
        print ("\rVocabulary built: %d" % len(words))
        self.words = words
        self.word_map = word_map  # Mapping from each token to its index in vocab   
        self.total = tot 
        self.maxx = maxx

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.word_map

    def indices(self, tokens):
        return [self.word_map[token] for token in tokens]

    def indice(self, token):
        return self.word_map[token]