import sys
sys.path.append("..")
from transformers import BertModel,BertConfig,BertTokenizer
from transformers.modeling_bert import BertEmbeddings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
import numpy as np
import operator
from model.dataset import BertDataset,construct

class MaskedBert(nn.Module):
    def __init__(self,bert,tokenizer,device,device_ids):
        super(MaskedBert,self).__init__()
        self.bert=bert
        self.device=device
        self.tokenizer=tokenizer

    def forward(self,inputs):
        input_embeds=inputs['origin_embeds']
        h=self.bert(**inputs)
        out=h.last_hidden_state.to(self.device)+input_embeds
        return out

class BiGRU(nn.Module):
    def __init__(self, embedding_size, hidden, n_layers, dropout=0.0):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(embedding_size, hidden, num_layers=n_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden*2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        prob = self.sigmoid(self.linear(out))
        return prob

class SoftMaskedBert(nn.Module):
    def __init__(self,bert,tokenizer,device,device_ids,gru_hidden=256,gru_layers=1,dropout=0.0):
        super(SoftMaskedBert,self).__init__()
        self.device=device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert=bert.to(device)
        # self.maskedbert=MaskedBert(bert,device,device_ids).to(device)
        # self.maskedbert=nn.DataParallel(self.maskedbert,device_ids)
        self.rnn = nn.GRU(embedding_size, gru_hidden, num_layers=gru_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.rnn.flatten_parameters()
        self.sigmoid = nn.Sigmoid().to(device)
        self.gru_linear = nn.Linear(gru_hidden * 2, 1)
        # self.bigru=BiGRU(embedding_size,gru_hidden,gru_layers).to(device)
        # self.bigru=nn.DataParallel(self.bigru,device_ids)

        bert_embedding=bert.embeddings
        word_embeddings_weight=bert.embeddings.word_embeddings.weight
        embeddings=nn.Parameter(word_embeddings_weight,True)
        bert_embedding.word_embeddings=nn.Embedding(self.config.vocab_size,embedding_size,_weight=embeddings)
        self.linear=nn.Linear(embedding_size,self.config.vocab_size)
        self.linear.weight=embeddings

        self.embedding = bert_embedding.to(device)
        self.tokenizer=tokenizer
        # mask_embed = self.embedding(torch.tensor([[tokenizer.mask_token_id]]).to(device))
        # self.mask_e = mask_embed.detach()
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.criterion_d=nn.BCELoss()
        # self.criterion_c=nn.NLLLoss()

    def forward(self,input_ids,input_tyi,input_attn_mask):
        # inputs=self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
        # outputs=self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(self.device)
        input_embeds = self.embedding(input_ids=input_ids)
        mask_e=self.embedding(torch.tensor([[self.tokenizer.mask_token_id]]).to(self.device)).detach()
        self.rnn.flatten_parameters()
        gru_out, _ = self.rnn(input_embeds)
        prob = self.sigmoid(self.gru_linear(gru_out))
        # prob=self.bigru(input_embeds)
        # label = (input_ids != output_ids) + 0
        # d_loss=self.criterion_d(prob.reshape(-1, prob.shape[1]),label.float())
        p=prob.reshape(prob.shape[0],prob.shape[1],1)
        input_embeds_ = (1 - p) * input_embeds + p * mask_e
        # inputs['inputs_embeds'] = input_embeds_
        # del inputs['input_ids']
        # out=self.maskedbert(inputs)
        h = self.bert(attention_mask=input_attn_mask,token_type_ids=input_tyi,inputs_embeds=input_embeds_)
        out = h.last_hidden_state + input_embeds
        out=self.softmax(self.linear(out))
        # c_loss=self.criterion_c(out.transpose(1,2),output_ids)
        # loss=0.2*d_loss+0.8*c_loss
        return prob,out