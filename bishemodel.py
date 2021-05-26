from transformers import BertTokenizer,BertModel
import os
import sys
import torch
import torch.nn as nn

class bertmodel(nn.Module):
    def __init__(self):
        super(bertmodel,self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert_drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768*3,4)
        self.out = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 2)
    def forward(self,ids,mask,token_type_ids,targets,device,fenci_start,fenci_end):
        tmp = torch.DoubleTensor().to(device)
        label = torch.Tensor().to(device)
        out1= self.bert(ids,attention_mask = mask ,token_type_ids = token_type_ids )
        out1 = out1.last_hidden_state
        out2 = self.bert_drop(out1)


        fin_in = torch.cat([out2,fenci_start,fenci_end],dim = 2)
        out3 = self.linear(fin_in)
        length = torch.sum(mask,dim = 1)
        for i in range(len(mask)):
            tmp = torch.cat([tmp,out3[i,:length[i],:]],dim = 0)
            label = torch.cat([label,targets[i,:length[i]]],dim = 0)
        #targets = targets.view(-1)
        label = label.long()
        out = self.out(tmp,label)
        return out
    def get_score(self,ids,mask,token_type_ids,fenci_start,fenci_end):
        out1 = self.bert(ids,attention_mask = mask ,token_type_ids = token_type_ids )
        out1 = out1.last_hidden_state
        out2 = self.bert_drop(out1)
        fin_in = torch.cat([out2,fenci_start,fenci_end],dim = 2)
        out3 = self.linear(fin_in)
        result = self.softmax(out3)
        result = torch.argmax(result,dim = 2)
        return result