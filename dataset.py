import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import random
from tqdm import tqdm


class BERTDataset(Dataset):
    def __init__(self,args,mode='train',remove_pkl=False):

        if remove_pkl:
            os.remove('{}/{}'.format(args.base_dir,mode))
        

        self.mode = mode
        self.args = args
        self.data = self.prepare_dataset()
        self.data['convert_data'] = []

        for idx in tqdm(range(self.data['len'])):
            s1,s2,task_label = self.data['data'][idx][0],self.data['data'][idx][1],self.data['label'][idx]
            self.data['convert_data'].append([self.convert_sentence(s1),self.convert_sentence(s2),task_label])


    def handl_pickle(self,data=None):

        if data is not None:
            with open('{}/{}'.format(self.args.base_dir,self.mode),'wb') as f:
                pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
            return data
        else:
            with open('{}/{}'.format(self.args.base_dir,self.mode),'rb') as f:
                return pickle.load(f)


    def prepare_dataset(self):

        filename = '{}/{}'.format(self.args.base_dir,self.mode)

        if not os.path.isfile(filename):
            print('Make data pkl file : {}'.format(self.mode))

            data = pd.read_csv('{}/{}.tsv'.format(self.args.base_dir,self.mode),delimiter='\t',header=0)
            data = data.dropna()

            pkl_data = {'data':[],'label':[],'len':0}
            labels = {'not_entailment':0,'entailment':1}

            for idx,s1,s2,label in zip(data['index'],data['sentence1'],data['sentence2'],data['label']):
                pkl_data['data'].append([s1,s2])
                pkl_data['label'].append([labels[label]])
                pkl_data['len'] += 1

            return self.handl_pickle(pkl_data)
        else:
            print('Ready to {}'.format(self.mode))
            return self.handl_pickle()


    def __len__(self):
        return self.data['len']

    def random_sentence(self,s1,s2):
        
        if random.random() < self.args.nsp_ratio:
            return s1,s2,1
        random_index = random.randrange(self.data['len'])
        return s1,self.data['convert_data'][random_index][1],0


    def convert_sentence(self,sentence):

        words,data = sentence.split(),[]

        for word in words:
            data.append(self.args.vocab['stoi'][word])
        return data


    def random_mask(self,words):
        data,label = [],[]

        for word in words:
            if random.random() < self.args.mlm_ratio:
                rand = random.random()
                if  rand < 0.8:
                    data_token = self.args.vocab['stoi']['<MASK>']
                elif rand < 0.9:
                    data_token = random.randrange(len(self.args.vocab['stoi'].values()))
                else:
                    data_token = word

                label_token = word
            else:
                data_token,label_token = word,word

            data.append(data_token)
            label.append(label_token)

        return data,label


    def __getitem__(self,idx):

        s1, s2, task_label = self.data['convert_data'][idx]

        if self.args.task:
            rm_s1, rm_s2 = s1, s2
            rm_s1_label, rm_s2_label = s1, s2
            rs_label = None
        else:
            rs_s1, rs_s2, rs_label = self.random_sentence(s1, s2)
            rm_s1, rm_s1_label = self.random_mask(rs_s1)
            rm_s2, rm_s2_label = self.random_mask(rs_s2)

        segment = [1 for _ in range(len(rm_s1))] + [2 for _ in range(len(rm_s2))]
        data = [self.args.vocab['stoi']['<CLS>']] + rm_s1 + [self.args.vocab['stoi']['<SEP>']] + rm_s2 + [self.args.vocab['stoi']['<SEP>']]
        label = [self.args.vocab['stoi']['<CLS>']] + rm_s1_label + [self.args.vocab['stoi']['<SEP>']] + rm_s2_label + [self.args.vocab['stoi']['<SEP>']]

        segment = segment[:self.args.max_len]
        data = data[:self.args.max_len]
        label = label[:self.args.max_len]

        padding = [self.args.vocab['stoi']['<PAD>'] for _ in range(self.args.max_len - len(data))]
        segment += [self.args.vocab['stoi']['<PAD>'] for _ in range(self.args.max_len - len(segment))]
        data += padding
        label += padding

        data = torch.tensor(data)
        segment = torch.tensor(segment)
        if self.mode == 'test':
            return {
                'data': data,
                'segment': segment
            }

        label = torch.tensor(label)
        return {
            'data': data,
            'segment': segment,
            'label': label,
            'rs_label': rs_label,
            'task_label': task_label
        }








