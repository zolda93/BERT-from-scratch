from torch.utils.data import DataLoader
from Vocab import *
from dataset import*


def get_dataloader(args):

    vocab = Vocab(args,filename=['train.tsv', 'dev.tsv'])
    args.vocab = vocab.get_data()

    train_data = BERTDataset(args, mode='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = BERTDataset(args, mode='dev')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader

