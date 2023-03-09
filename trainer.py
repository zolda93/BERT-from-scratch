import torch
import torch.nn.functional as F

import os
from sklearn.metrics import f1_score, precision_score, recall_score


class Trainer:

    def __init__(self,
                args,
                model,
                optimizer,
                criterion):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


    def metrics(self,y,y_pred,average='weighted'):
        f1 = f1_score(y,y_pred,average=average)
        precision = precision_score(y,y_pred,average=average)
        recall = recall_score(y,y_pred,average=average)

        return f1,precision,recall_score

    def save_checkpoint(self,epoch):
        print('Model Saving...')

        
        model_state_dict = self.model.state_dict()


        torch.save({
            'model_state_dict':model_state_dict,
            'global_epoch':epoch,
            'optimzier_state_dict':self.optimizer.state_dict()},
            os.path.join('checkpoints','checkpoint_model_bedt.pth'))


    def train(self,epoch,train_loader):
        self.model.train()

        mlm_acc, nsp_acc, mlm_losses, nsp_losses, losses, step = 0., 0., 0., 0., 0., 1
        total = 0.

        for idx,data in enumerate(train_loader):
            inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['rs_label']

            if self.args.task:
                inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['task_label']

            if self.args.cuda:
                inputs, segment, mlm_label, nsp_label = inputs.cuda(), segment.cuda(), mlm_label.cuda(), nsp_label.cuda()

            mlm_logits,nsp_logits = self.model(inputs,segment)
            mlm_logits = mlm_logits.view(-1,self.args.vocab_len)
            mlm_label = mlm_label.view(-1)

            self.optimizer.zero_grad()
            nsp_loss = self.criterion['nsp'](nsp_logits,nsp_label)
            nsp_losses += nsp_loss.item()

            if not self.args.task:
                mlm_loss = self.criterion['mlm'](mlm_logits,mlm_label)
                mlm_losses += mlm_loss.item()
                loss = mlm_loss + nsp_loss
            else:
                loss = nsp_loss

            losses += loss.item()
            loss.backward()
            self.optimizer.step()

            mlm_pred = F.softmax(mlm_logits,dim=-1).max(-1)[1]
            nsp_pred = F.softmax(nsp_logits,dim=-1).max(-1)[1]
            
            inds = (mlm_label != 0).view(-1)
            mlm_acc += mlm_pred[inds].eq(mlm_label[inds]).sum().item()
            nsp_acc += nsp_pred.eq(nsp_label).sum().item()

            step +=1
            total += inds.size(0)

        mlm_losses /= step
        nsp_losses /= step
        losses /= step
        mlm_acc = mlm_acc / total * 100.
        nsp_acc = nsp_acc / len(train_loader.dataset) * 100.
        print('[Train Epoch: {0:4d}] mlm loss: {1:.3f}, nsp loss: {2:.3f}, loss: {3:.3f}, mlm acc: {4:.4f}, nsp acc: {5:.4f}'.format(epoch, mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc))

        return mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc


    def eval(self,epoch,test_loader):
        self.model.eval()

        mlm_acc, nsp_acc, mlm_losses, nsp_losses, losses, step = 0., 0., 0., 0., 0., 0.
        total = 0.
        
        with torch.no_grad():
            for data in test_loader:
                inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['rs_label']

                if self.args.task:
                    inputs, segment, mlm_label, nsp_label = data['data'], data['segment'], data['label'], data['task_label']

                if self.args.cuda:
                    inputs, segment, mlm_label, nsp_label = inputs.cuda(), segment.cuda(), mlm_label.cuda(), nsp_label.cuda()

                mlm_logits,nsp_logits = self.model(inputs,segment)
                mlm_logits = mlm_logits.view(-1,self.args.vocab_len)
                mlm_label = mlm_label.view(-1)

                nsp_loss = self.criterion['nsp'](nsp_logits,nsp_label)
                nsp_losses += nsp_loss.item()

                if self.args.task:
                    mlm_loss = self.criterion['mlm'](mlm_logits,mlm_label)
                    mlm_losses += mlm_loss.item()
                    loss = nsp_loss + mlm_loss
                else:
                    loss = nsp_loss

                losses += loss.item()

                mlm_pred = F.softmax(mlm_logits,dim=-1).max(-1)[1]
                nsp_pred = F.softmax(nsp_logits,dim=-1).max(-1)[1]


                inds = (mlm_label != 0).view(-1)
                mlm_acc += mlm_pred[inds].eq(mlm_label[inds]).sum().item()
                nsp_acc += nsp_pred.eq(nsp_label).sum().item()

                step += 1
                total += inds.size(0)

            mlm_losses /= step
            nsp_losses /= step
            losses /= step
            mlm_acc = mlm_acc / total * 100.
            nsp_acc = nsp_acc / len(test_loader.dataset) * 100.
            print('[Test Epoch: {0:4d}] mlm loss: {1:.3f}, nsp loss: {2:.3f}, loss: {3:.3f}, mlm acc: {4:.4f}, nsp acc: {5:.4f}'.format(epoch, mlm_losses,nsp_losses, losses,mlm_acc, nsp_acc))
            return mlm_losses, nsp_losses, losses, mlm_acc, nsp_acc


























