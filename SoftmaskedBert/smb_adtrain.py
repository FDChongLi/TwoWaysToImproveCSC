# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch
import numpy as np
import argparse
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import operator
from model import SoftMaskedBert,construct,BertDataset,logitGen,readAllConfusionSet
import os

class Trainer():
    def __init__(self,smb,optimizer,tokenizer,device):
        self.model=smb
        self.optim=optimizer
        self.tokenizer=tokenizer
        self.criterion_d=nn.BCELoss()
        self.criterion_c=nn.NLLLoss()
        self.device=device
        self.confusion_set=readAllConfusionSet('save/allSim2.file')

    def train(self,train):
        self.model.train()
        total_loss=0
        for batch in train:
            inputs=self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs=self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            input_ids,input_tyi,input_attn_mask=inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
            output_ids,output_tyi,output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask']
            prob,out=self.model(input_ids,input_tyi,input_attn_mask)
            label = (input_ids != output_ids) + 0
            d_loss = self.criterion_d(prob.reshape(-1, prob.shape[1]), label.float())
            c_loss=self.criterion_c(out.transpose(1,2),output_ids)
            loss=0.2*d_loss+0.8*c_loss
            total_loss+=loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return total_loss

    def ad_train(self,train,ratio=0.15):
        self.model.train()
        total_loss = 0
        count=1
        for batch in train:
            if(count%3!=0):
                inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(
                    self.device)
                outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
                    self.device)
                input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                    'attention_mask']
                output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                    'attention_mask']
                prob, out = self.model(input_ids, input_tyi, input_attn_mask)
                label = (input_ids != output_ids) + 0
                d_loss = self.criterion_d(prob.reshape(-1, prob.shape[1]), label.float())
                c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
                loss = 0.2 * d_loss + 0.8 * c_loss
                total_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                ad_sample_gen, _, _ = logitGen(batch['output'],batch['output'], self.model, self.tokenizer, self.device, self.confusion_set,ratio)
                inputs = []
                outputs = []
                for i in range(len(ad_sample_gen)):
                    inputs.append(ad_sample_gen[i]['input'])
                    outputs.append(ad_sample_gen[i]['output'])
                inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.tokenizer(outputs, padding=True, truncation=True, return_tensors="pt").to(
                    self.device)
                input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                    'attention_mask']
                output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                    'attention_mask']
                prob, out = self.model(input_ids, input_tyi, input_attn_mask)
                label = (input_ids != output_ids) + 0
                d_loss = self.criterion_d(prob.reshape(-1, prob.shape[1]), label.float())
                c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
                loss = 0.2 * d_loss + 0.8 * c_loss
                total_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            count+=1


    def test(self,test):
        self.model.eval()
        total_loss=0
        for batch in test:
            inputs=self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs=self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            input_ids,input_tyi,input_attn_mask=inputs['input_ids'],inputs['token_type_ids'],inputs['attention_mask']
            output_ids,output_tyi,output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask']
            prob,out=self.model(input_ids,input_tyi,input_attn_mask)
            label = (input_ids != output_ids) + 0
            d_loss = self.criterion_d(prob.reshape(-1, prob.shape[1]), label.float())
            c_loss=self.criterion_c(out.transpose(1,2),output_ids)
            loss=0.2*d_loss+0.8*c_loss
            total_loss+=loss.item()
        return total_loss


    def save(self,name):
        if (isinstance(self.model, nn.DataParallel)):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def load(self,name):
        self.model.load_state_dict(torch.load(name))

    def testSet(self,test):
        self.model.eval()
        sen_acc = 0
        setsum = 0
        sen_mod = 0
        sen_mod_acc = 0
        sen_tar_mod = 0
        d_sen_acc = 0
        d_sen_mod = 0
        d_sen_mod_acc = 0
        d_sen_tar_mod = 0
        for batch in test:
            inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
                self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            prob, out = self.model(input_ids, input_tyi, input_attn_mask)
            out = out.argmax(dim=-1)
            mod_sen = [not out[i].equal(input_ids[i]) for i in range(len(out))]  # 修改过的句子
            acc_sen = [out[i].equal(output_ids[i]) for i in range(len(out))]  # 正确的句子
            tar_sen = [not output_ids[i].equal(input_ids[i]) for i in range(len(output_ids))]  # 应该修改的句子
            sen_mod += sum(mod_sen)
            sen_mod_acc += sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))
            sen_tar_mod += sum(tar_sen)
            sen_acc += sum([out[i].equal(output_ids[i]) for i in range(len(out))])
            setsum += output_ids.shape[0]

            prob_ = [[0 if out[i][j] == input_ids[i][j] else 1 for j in range(len(out[i]))] for i in
                     range(len(out))]
            label = [[0 if input_ids[i][j] == output_ids[i][j] else 1 for j in
                      range(len(input_ids[i]))] for i in range(len(input_ids))]
            d_acc_sen = [operator.eq(prob_[i], label[i]) for i in range(len(prob_))]
            d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]
            d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]
            d_sen_mod += sum(d_mod_sen)
            d_sen_mod_acc += sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))
            d_sen_tar_mod += sum(d_tar_sen)
            d_sen_acc += sum(d_acc_sen)
        d_precision = d_sen_mod_acc / d_sen_mod
        d_recall = d_sen_mod_acc / d_sen_tar_mod
        d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall)
        c_precision = sen_mod_acc / sen_mod
        c_recall = sen_mod_acc / sen_tar_mod
        c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall)
        print("detection sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc / setsum, d_precision,
                                                                                       d_recall, d_F1))
        print("correction sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(sen_acc / setsum,
                                                                                        sen_mod_acc / sen_mod,
                                                                                        sen_mod_acc / sen_tar_mod,
                                                                                        c_F1))
        print("sentence target modify:{0},sentence sum:{1},sentence modified accurate:{2}".format(sen_tar_mod, setsum,
                                                                                                  sen_mod_acc))

        # accuracy, precision, recall, F1
        return sen_acc / setsum, sen_mod_acc / sen_mod, sen_mod_acc / sen_tar_mod, c_F1

    def attack(self,test,ratio=0.15,batch_size=20):
        self.model.eval()
        ad_gen = []
        all_count=0
        all_mod_count=0
        for batch in test:
            gen, count, mod_count = logitGen(batch['output'],batch['output'], self.model, tokenizer, device, self.confusion_set,ratio)
            ad_gen += gen
            all_count+=sum(count)
            all_mod_count+=mod_count
        ad = BertDataset(tokenizer, ad_gen)
        ad = DataLoader(ad, batch_size=batch_size)
        # success = [0] * 30  # 攻击成功
        success_num=0
        # fail = [0] * 30  # 攻击失败
        fail_num=0
        for batch in ad:
            inputs = tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            prob, out = self.model(input_ids, input_tyi, input_attn_mask)
            out = out.argmax(dim=-1)
            for i in range(len(batch['input'])):
                if (out[i].equal(output_ids[i])):
                    # fail[batch['mod'][i]] += 1
                    fail_num+=1
                else:
                    # success[batch['mod'][i]] += 1
                    success_num+=1
        success_rate=success_num/(success_num+fail_num)
        # print(success)
        # print(fail)
        print("ratio:{0},success num:{1},fail num:{2},success rate:{3}".format(ratio,success_num,fail_num,success_rate))
        self.testSet(ad)
        return all_count,all_mod_count

if __name__ == "__main__":
    import time

    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--load_model', type=str2bool, nargs='?', const=False)
    parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')

    parser.add_argument('--do_train', type=str2bool, nargs='?', const=False)
    parser.add_argument('--train_data', type=str, default='../data/13train.txt')
    parser.add_argument('--do_valid', type=str2bool, nargs='?', const=False)
    parser.add_argument('--valid_data', type=str, default='../data/13valid.txt')
    parser.add_argument('--do_test', type=str2bool, nargs='?', const=False)
    parser.add_argument('--test_data', type=str, default='../data/13test.txt')

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--do_save', type=str2bool, nargs='?', const=False)
    parser.add_argument('--save_dir', type=str, default='../save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argumetn('--train_ratio',type=float, default=0.0)
    parser.add_argument('--attack_ratio',type=float,default=0.0)

    args = parser.parse_args()
    task_name = args.task_name
    print("----Task: " + task_name + " begin !----")

    setup_seed(int(args.seed))
    start = time.time()

    # device_ids=[0,1]
    device_ids = [i for i in range(int(args.gpu_num))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert = BertModel.from_pretrained('bert-base-chinese', return_dict=True)
    # embedding = bert.embeddings.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    config = BertConfig.from_pretrained('bert-base-chinese')

    model = SoftMaskedBert(bert, tokenizer, device, device_ids).to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    model = nn.DataParallel(model, device_ids)

    if args.do_train:
        train = construct(args.train_data)
        train = BertDataset(tokenizer, train)
        train = DataLoader(train, batch_size=int(args.batch_size), shuffle=True)

    if args.do_valid:
        valid = construct(args.valid_data)
        valid = BertDataset(tokenizer, valid)
        valid = DataLoader(valid, batch_size=int(args.batch_size), shuffle=True)

    if args.do_test:
        test = construct(args.test_data)
        test = BertDataset(tokenizer, test)
        test = DataLoader(test, batch_size=int(args.batch_size), shuffle=True)

    optimizer = Adam(model.parameters(), float(args.learning_rate))
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    trainer = Trainer(model, optimizer, tokenizer, device)
    # max_f1 = 0
    # best_epoch = 0

    if args.do_train:
        for e in range(int(args.epoch)):
            train_loss = trainer.ad_train(train,args.train_ratio)

            if args.do_valid:
                trainer.testSet(valid)
                trainer.attack(valid,args.attack_ratio,args.batch_size)
            print(task_name, ",epoch {0},train_loss:{1}".format(e + 1, train_loss))

            # best_epoch = e + 1
            if args.do_save:
                model_save_path = args.save_dir + '/epoch{0}.pkl'.format(e + 1)
                trainer.save(model_save_path)
                print("save model done!")
            print("Time cost:", time.time() - start, "s")
            print("-" * 10)

        # model_best_path = args.save_dir + '/epoch{0}.pkl'.format(best_epoch)
        # model_save_path = args.save_dir + '/model.pkl'

        # copy the best model to standard name
        # os.system('cp ' + model_best_path + " " + model_save_path)

    if args.do_test:
        trainer.testSet(test)
        trainer.attack(test, args.attack_ratio, args.batch_size)