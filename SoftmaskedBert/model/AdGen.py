import pickle
import random
import torch
import numpy as np

def getConfusionSet(relationFilePath,relAddSet):
    ret = dict()
    relationFileHandle = open(relationFilePath,"r",encoding='UTF-8')
    while True:
        line = relationFileHandle.readline()
        if not line :
            break
        e1, e2, rel = line.strip().split("|")
        if rel in relAddSet:
            # initial the dictionary
            if e1 not in ret:
                ret.setdefault(e1,set())
            ret[e1].add(e2)

            # initial the dictionary
            if e2 not in ret:
                ret.setdefault(e2,set())
            ret[e2].add(e1)

    relationFileHandle.close()
    return ret

def generateConfusionSet(relationFilePath='../save/spellGraphs.txt'):
    relationlist=["同音同调", "同音异调","近音同调", "近音异调","同部首同笔画", "形近"]
    confusionSet=getConfusionSet(relationFilePath,set(relationlist))
    with open("../save/allSim2.file", "wb") as f_:
        pickle.dump(confusionSet, f_)
    relationlist1 = ["同音同调", "同音异调"]
    pronSame=getConfusionSet(relationFilePath, set(relationlist1))
    with open("../save/pronSame2.file", "wb") as f_:
        pickle.dump(pronSame, f_)

    # diction of similar pronuciation
    relationlist2 = ["近音同调", "近音异调"]
    pronSim = getConfusionSet(relationFilePath, set(relationlist2))
    with open("../save/pronSim2.file", "wb") as f_:
        pickle.dump(pronSim, f_)

    # diction of similar characters
    relationlist3 = ["同部首同笔画", "形近"]
    shapeSim = getConfusionSet(relationFilePath, set(relationlist3))
    with open("../save/shapeSim2.file", "wb") as f_:
        pickle.dump(shapeSim, f_)

def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim=pickle.load(f)
        return allSim

def readThreeConfusionSet(dirpath):
    pronSamePath=dirpath+"/pronSame2.file"
    with open(pronSamePath,'rb') as f:
        pronSame=pickle.load(f)
    pronSimPath=dirpath+"/pronSim2.file"
    with open(pronSimPath,'rb') as f:
        pronSim=pickle.load(f)
    shapeSimPath=dirpath+"/shapeSim2.file"
    with open(shapeSimPath,'rb') as f:
        shapeSim=pickle.load(f)
    return pronSame,pronSim,shapeSim

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def chooseLargest(char,prob,tokenizer,confusionSet):
    confusion_set=confusionSet.get(char,set())
    max_prob=None
    max_cand=None
    for cand in confusion_set:
        cand_id=tokenizer(cand)['input_ids'][1]
        if(max_prob==None or prob[cand_id]>max_prob):
            max_prob=prob[cand_id]
            max_cand=cand
    return max_cand

def logitGen(input_batch,output_batch,model,tokenizer,device,confusionSet,ratio=0.15,rerank=True):
    new_input_set=input_batch
    new_output_set=output_batch
    set_tokens=[]
    ad_sample_gen=[]
    logit_rank=None
    count=[0]*len(input_batch)#每句话错字个数
    mod_count=[0]*len(input_batch)#每句话修改次数
    count_mod=0#总修改字数
    logit_idx=[0]*len(input_batch)#每句话从高到低排序到第几个字
    mod_len=[0]*len(input_batch)#可以修改的数量
    seq_len=[0]*len(input_batch)#句子padding前长度
    while len(new_input_set)!=0:
        inputs = tokenizer(new_input_set, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs =tokenizer(new_output_set, padding=True, truncation=True, return_tensors="pt").to(device)
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
        'attention_mask']
        prob, out = model(input_ids, input_tyi, input_attn_mask)#out:[batch_size,seq_len,vocab_size]

        if logit_rank==None or rerank:#是否每次替换后重新计算
            prob_rank=out.argsort(dim=-1)#vocab_size维度上的排序,从小到大
            prob_dif=[]#[batch_size,seq_len]
            for i in range(out.shape[0]):
                seq_len[i]=sum(input_attn_mask[i])-2#开头，结尾
                set_prob_dif=[]
                for j in range(out.shape[1]):#保持长度一致
                    set_prob_dif.append(out[i][j][prob_rank[i][j][-1]]-out[i][j][prob_rank[i][j][-2]])
                prob_dif.append(set_prob_dif)
                mod_len[i]=int(ratio*seq_len[i])
                set_tokens.append(tokenizer.tokenize(new_input_set[i]))

            logit_rank=torch.tensor(prob_dif).argsort(dim=-1).numpy()
            logit_idx = [0] * out.shape[0]

        # for i in range(out.shape[0]):
        #     count[i] = sum([input_ids[i][j] != output_ids[i][j] for j in range(len(input_ids[i]))])

        out_ids=out.argmax(dim=-1)#[batch_size,seq_len]
        new_input_set_=[]
        new_output_set_=[]
        logit_rank_=[]
        set_tokens_=[]
        mod_count_=[]
        logit_idx_=[]
        mod_len_=[]

        for i in range(len(out_ids)):#batch_size
            is_added=False
            if(out_ids[i].equal(output_ids[i]) and mod_count[i]<mod_len[i] and not is_added):
                input_token=tokenizer.tokenize(new_input_set[i])
                output_token=tokenizer.tokenize(new_output_set[i])
                sub_idx=logit_rank[i][logit_idx[i]]-1
                while sub_idx>=len(input_token) or (not is_Chinese(input_token[sub_idx])) or input_token[sub_idx]!=output_token[sub_idx]:
                    logit_idx[i]+=1
                    if(logit_idx[i]>=out.shape[1]):
                        ad_sample_gen.append({'input':''.join(set_tokens[i]).replace("##",""),'output':new_output_set[i],'mod':mod_count[i]})
                        is_added=True
                        break
                    sub_idx=logit_rank[i][logit_idx[i]]-1
                if(is_added):
                    continue
                sub_cand=chooseLargest(output_token[sub_idx],out[i][sub_idx],tokenizer,confusionSet)
                while sub_cand==None:
                    logit_idx[i]+=1
                    if (logit_idx[i] >= out.shape[1]):
                        ad_sample_gen.append({'input':''.join(set_tokens[i]).replace("##",""),'output':new_output_set[i],'mod':mod_count[i]})
                        is_added=True
                        break
                    sub_idx = logit_rank[i][logit_idx[i]]-1
                    if(sub_idx>=len(input_token)):
                        continue
                    sub_cand = chooseLargest(output_token[sub_idx], out[i][sub_idx], tokenizer,confusionSet)
                if(is_added):
                    continue
                input_token[sub_idx]=sub_cand
                set_tokens[i][sub_idx]=sub_cand
                mod_count[i]+=1
                new_input_set_.append(''.join(input_token).replace("##",""))
                new_output_set_.append(new_output_set[i])
                logit_rank_.append(logit_rank[i])
                set_tokens_.append(set_tokens[i])
                mod_count_.append(mod_count[i])
                logit_idx_.append(logit_idx[i])
                mod_len_.append(mod_len[i])
            else:
                ad_sample_gen.append({'input':''.join(set_tokens[i]).replace("##",""),'output':new_output_set[i],'mod':mod_count[i]})
                count_mod+=mod_count[i]

        new_input_set=new_input_set_
        new_output_set=new_output_set_
        logit_rank=logit_rank_
        set_tokens=set_tokens_
        mod_count=mod_count_
        logit_idx=logit_idx_
        mod_len=mod_len_

        if(len(new_input_set)==0):
            for i in range(out.shape[0]):
                count[i] = sum([input_ids[i][j] != output_ids[i][j] for j in range(len(input_ids[i]))])

    return ad_sample_gen,count,count_mod

def BFTLogitGen(input_batch,output_batch,model,tokenizer,device,confusionSet,ratio=0.15,rerank=True):
    new_input_set = input_batch
    new_output_set = output_batch
    set_tokens = []  # 解决UNK问题
    ad_sample_gen = []
    logit_rank = None
    count = [0] * len(input_batch)  # 每句话错字个数
    mod_count = [0] * len(input_batch)  # 每句话修改次数
    count_mod=0
    logit_idx = [0] * len(input_batch)  # 每句话从高到低排序到第几个字
    mod_len = [0] * len(input_batch)  # 可以修改的数量
    seq_len = [0] * len(input_batch)  # 句子padding前长度
    while len(new_input_set) != 0:
        inputs = tokenizer(new_input_set, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = tokenizer(new_output_set, padding=True, truncation=True, return_tensors="pt").to(device)
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
            'attention_mask']
        out = model(input_ids, input_tyi, input_attn_mask)

        if logit_rank == None or rerank:
            prob_rank = out.argsort(dim=-1)  # vocab_size维度上的排序,从小到大
            prob_dif = []  # [batch_size,seq_len]
            for i in range(out.shape[0]):
                seq_len[i] = sum(input_attn_mask[i]) - 2  # 开头，结尾
                set_prob_dif = []
                for j in range(out.shape[1]):  # 保持长度一致
                    set_prob_dif.append(out[i][j][prob_rank[i][j][-1]] - out[i][j][prob_rank[i][j][-2]])
                prob_dif.append(set_prob_dif)
                mod_len[i] = int(ratio * seq_len[i])
                set_tokens.append(tokenizer.tokenize(new_input_set[i]))

            logit_rank = torch.tensor(prob_dif).argsort(dim=-1).numpy()
            logit_idx = [0] * out.shape[0]

        # for i in range(out.shape[0]):
        #     count[i] = sum([input_ids[i][j] != output_ids[i][j] for j in range(len(input_ids[i]))])

        out_ids = out.argmax(dim=-1)  # [batch_size,seq_len]
        new_input_set_ = []
        new_output_set_ = []
        logit_rank_ = []
        set_tokens_ = []
        mod_count_ = []
        logit_idx_ = []
        mod_len_ = []

        for i in range(len(out_ids)):  # batch_size
            is_added = False
            if (out_ids[i].equal(output_ids[i]) and mod_count[i] < mod_len[i] and not is_added):
                input_token = tokenizer.tokenize(new_input_set[i])
                output_token = tokenizer.tokenize(new_output_set[i])
                sub_idx = logit_rank[i][logit_idx[i]]-1
                while sub_idx >= len(input_token) or (not is_Chinese(input_token[sub_idx])) or input_token[sub_idx] != \
                        output_token[sub_idx]:
                    logit_idx[i] += 1
                    if (logit_idx[i] >= out.shape[1]):
                        ad_sample_gen.append(
                            {'input': ''.join(set_tokens[i]).replace("##", ""), 'output': new_output_set[i],
                             'mod': mod_count[i]})
                        is_added = True
                        break
                    sub_idx = logit_rank[i][logit_idx[i]]-1
                if (is_added):
                    continue
                sub_cand = chooseLargest(output_token[sub_idx], out[i][sub_idx], tokenizer, confusionSet)
                while sub_cand == None:
                    logit_idx[i] += 1
                    if (logit_idx[i] >= out.shape[1]):
                        ad_sample_gen.append(
                            {'input': ''.join(set_tokens[i]).replace("##", ""), 'output': new_output_set[i],
                             'mod': mod_count[i]})
                        is_added = True
                        break
                    sub_idx = logit_rank[i][logit_idx[i]]-1
                    if (sub_idx >= len(input_token)):
                        continue
                    sub_cand = chooseLargest(output_token[sub_idx], out[i][sub_idx], tokenizer, confusionSet)
                if (is_added):
                    continue
                input_token[sub_idx] = sub_cand
                set_tokens[i][sub_idx] = sub_cand
                mod_count[i] += 1
                new_input_set_.append(''.join(input_token).replace("##", ""))
                new_output_set_.append(new_output_set[i])
                logit_rank_.append(logit_rank[i])
                set_tokens_.append(set_tokens[i])
                mod_count_.append(mod_count[i])
                logit_idx_.append(logit_idx[i])
                mod_len_.append(mod_len[i])
            else:
                ad_sample_gen.append({'input': ''.join(set_tokens[i]).replace("##", ""), 'output': new_output_set[i],
                                      'mod': mod_count[i]})
                count_mod += mod_count[i]

        new_input_set = new_input_set_
        new_output_set = new_output_set_
        logit_rank = logit_rank_
        set_tokens = set_tokens_
        mod_count = mod_count_
        logit_idx = logit_idx_
        mod_len = mod_len_

        if (len(new_input_set) == 0):
            for i in range(out.shape[0]):
                count[i] = sum([input_ids[i][j] != output_ids[i][j] for j in range(len(input_ids[i]))])

    return ad_sample_gen, count, count_mod