import sys
sys.path.append("..")
import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel,BertConfig,BertTokenizer
from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
from model.dataset import BertDataset,construct

class BertFineTune(nn.Module):
    def __init__(self,bert,tokenizer,device,device_ids):
        super(BertFineTune,self).__init__()
        self.device=device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert=bert.to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        bert_embedding=bert.embeddings
        word_embeddings_weight=bert.embeddings.word_embeddings.weight
        embeddings=nn.Parameter(word_embeddings_weight,True)
        bert_embedding.word_embeddings=nn.Embedding(self.config.vocab_size,embedding_size,_weight=embeddings)
        self.linear=nn.Linear(embedding_size,self.config.vocab_size)
        self.linear.weight=embeddings

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,input_ids,input_tyi,input_attn_mask):
        h = self.bert(input_ids=input_ids,token_type_ids=input_tyi,attention_mask=input_attn_mask)
        out=self.softmax(self.linear(h.last_hidden_state))
        return out