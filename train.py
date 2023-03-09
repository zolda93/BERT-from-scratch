

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from dataloader import get_dataloader
from config import load_args
from model import BERT
from trainer import Trainer


def train():

    args = load_args()
    
    train_loader,test_loader = get_dataloader(args)
    print(train_loader)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args.vocab_len = len(args.vocab['stoi'].keys())
    

    model = BERT(args.vocab_len,args.max_len,args.heads,args.embedding_dim,args.N)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion = {
      'mlm':nn.CrossEntropyLoss(),
      'nsp':nn.CrossEntropyLoss()
    }

    trainer = Trainer(args,model,optimizer,criterion)

    if args.cuda:
        model = model.cuda()

    if args.task:
        print('Start Down StreamTask')
        args.epochs = 3
        args.lr = 3e-5
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict['model_state_dict'])
        criterion['mlm'] = None

        for epoch in tqdm(range(1,args.epochs + 1)):
            train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = trainer.train(epoch,train_loader)
            train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = trainer.eval(epoch,test_loader)
            trainer.save_checkpoint(epoch)
    else:
        print('Start Pre-training...')

        for epoch in tqdm(range(1,args.epochs)):
          train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = trainer.train(epoch,train_loader)
          train_mlm_loss, train_nsp_loss, train_loss, train_mlm_acc, train_nsp_acc = trainer.eval(epoch,test_loader)
          trainer.save_checkpoint(epoch)




