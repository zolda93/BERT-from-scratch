import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,sentence_len=128):
        super().__init__()

        self.positionalencoding = torch.zeros((sentence_len,embedding_dim))

        for pos in range(0,sentence_len):
            for i in range(0,embedding_dim//2):
                self.positionalencoding[pos, 2 * i] = math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                self.positionalencoding[pos, 2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / embedding_dim))

        self.register_buffer('poeisitional_encoding',self.positionalencoding)


    def forward(self,x:Tensor)->Tensor:

        sentence_len = x.size(1)
        out = x + self.positionalencoding[:sentence_len,:].to(x)
        return out




class ScaledDotProductAttention(nn.Module):
    def __init__(self,dk):
        super().__init__()

        self.dk = dk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q:Tensor,k:Tensor,v:Tensor,mask:Optional[Tensor]=None)->Tensor:

        out = torch.matmul(q,k.transpose(2,3))
        out /= math.sqrt(self.dk)

        if mask is not None:
            out = out.masked_fill(mask==0,-1e9)

        out = self.softmax(out)
        out = torch.matmul(out,v)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self,head:int,embedding_dim:int,dropout_rate:float=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head = head
        self.dk = embedding_dim // head

        self.Q = nn.Linear(embedding_dim,embedding_dim)
        self.K = nn.Linear(embedding_dim,embedding_dim)
        self.V = nn.Linear(embedding_dim,embedding_dim)

        self.scores = ScaledDotProductAttention(dk=self.dk)
        self.dropout = nn.Dropout(dropout_rate)
        self.ff = nn.Linear(embedding_dim,embedding_dim)


    def forward(self,
                query:Tensor,
                key:Tensor,
                value:Tensor,
                mask:Optional[Tensor]=None)->Tensor:

        batch,_,embedding_dim = query.size()

        q_out = self.Q(query).view(batch,-1,self.head,self.dk).transpose(1,2)
        k_out = self.K(key).view(batch,-1,self.head,self.dk).transpose(1,2)
        v_out = self.V(value).view(batch,-1,self.head,self.dk).transpose(1,2)

        score = self.scores(q_out,k_out,v_out,mask).transpose(1,2).contiguous().view(batch,-1,embedding_dim)

        out = self.dropout(score)
        out = self.ff(out)

        return out


class LayerNorm(nn.Module):
    def __init__(self,embedding_dim,eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(embedding_dim),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(embedding_dim),requires_grad=True)


    def forward(self,x:Tensor)->Tensor:

        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        norm = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return norm



class FeedForward(nn.Module):
    def __init__(self,embedding_dim,dropout_rate=0.1):
        super().__init__()

        self.inner_dim = 2048
        self.ff = nn.Sequential(
                nn.Linear(embedding_dim, self.inner_dim),
                nn.Dropout(dropout_rate),
                nn.GELU(),
                nn.Linear(self.inner_dim, embedding_dim),
                nn.Dropout(dropout_rate)
                )


    def forward(self,x:Tensor)->Tensor:
        out = self.ff(x)
        return out



class EncoderLayer(nn.Module):
    def __init__(self,head,embedding_dim,dropout_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(head,embedding_dim,dropout_rate)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.ff = FeedForward(embedding_dim,dropout_rate)


    def forward(self,x,x_mask):
        mha_out = self.dropout1(self.mha(x,x,x,x_mask))
        mha_out = self.norm1(mha_out + x)

        ff_out = self.dropout2(self.ff(mha_out))
        ff_out = self.norm2(ff_out + mha_out)
        return ff_out


class StackedEncoderLayer(nn.Module):
    def __init__(self,input_size,max_len,heads,embedding_dim,N,dropout_rate=0.1):
        super().__init__()

        self.input_size = input_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.N = N
        stacked_encoder_layers = []


        self.token_embedding = nn.Embedding(input_size,embedding_dim)
        self.segment_embedding = nn.Embedding(3,embedding_dim)
        self.positonalencoding = PositionalEncoding(embedding_dim)

        for _ in range(N):
            stacked_encoder_layers.append(EncoderLayer(heads,embedding_dim,dropout_rate))


        self.encoder = nn.Sequential(*stacked_encoder_layers)


    def forward(self,x:Tensor,mask:Tensor,segment:Tensor)->Tensor:
        token_embedding = self.token_embedding(x)
        segment_embedding = self.segment_embedding(segment)
        out = token_embedding + segment_embedding + self.positonalencoding(token_embedding)

        for _ in range(self.N):
            out = self.encoder[_](out,mask)
        return out




class BERT(nn.Module):
    def __init__(self,input_vocab,max_len,heads,embedding_dim,N,dropout_rate=0.1):
        super().__init__()

        self.encoder = StackedEncoderLayer(input_vocab,max_len,heads,embedding_dim,N,dropout_rate)
        self.mlm_out = nn.Sequential(nn.Linear(embedding_dim,input_vocab))
        self.nsp_out = nn.Sequential(nn.Linear(embedding_dim,2))



    def forward(self,x:Tensor,segment:Tensor):

        mask = (x > 0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        out = self.encoder(x,mask,segment)
        return self.mlm_out(out),self.nsp_out(out[:,0])




































