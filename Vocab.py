from collections import Counter,defaultdict
import pandas as pd
import pickle
import os
from tqdm import tqdm

CLS  = '<CLS>'
SEP  = '<SEP>'
PAD  = '<PAD>'
END  = '<END>'
MASK = '<MASK>'




class Vocab:

    def __init__(self,args,filename=None):

        self.out_filename = '{}/corpus.pkl'.format(args.base_dir)

        if not os.path.isfile(self.out_filename):
            self.corpus = Counter()
            self.itos,self.stoi = defaultdict(int),defaultdict(str)
            self.itos[0], self.itos[1], self.itos[2], self.itos[3], self.itos[4] = PAD,CLS,SEP,END,MASK
            self.stoi[PAD], self.stoi[CLS], self.stoi[SEP], self.stoi[END], self.stoi[MASK] = 0, 1, 2, 3, 4
            self.cnt = 5


            for f in tqdm(filename):
                _data = pd.read_csv('{}/{}'.format(args.base_dir,f),delimiter='\t',header=0)

                for idx,s1,s2 in zip(_data['index'],_data['sentence1'],_data['sentence2']):
                    self.build_vocab(s1)
                    self.build_vocab(s2)

            data = {'corpus':self.corpus,'itos':self.itos,'stoi':self.stoi}

            with open(self.out_filename,'wb') as f:
                pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)

    def get_data(self):
        with open(self.out_filename,'rb') as f:
            return pickle.load(f)

    def build_vocab(self,sentence):

        words = sentence.split()

        for word in words:
            self.corpus[word] += 1

            if word not in self.stoi.keys():
                self.stoi[word] = self.cnt
                self.itos[self.cnt] = word
                self.cnt += 1




