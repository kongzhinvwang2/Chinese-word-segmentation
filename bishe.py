
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import bishemodel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer,BertModel,BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import psutil
import os
import pynvml

fenci_model = torch.load("/home/hsl/hslcode/squad_md.pth")
fenci_model.eval()
device = torch.device("cuda")
fenci_model.to(device)

info = psutil.virtual_memory()

def f1(answer,result):
    num_pre = 0
    res = []
    num_word = 0
    for i in range(len(answer)):
        for j in answer[i]:
            num_word+=1

    for i in result:
        region = []
        start = 0
        for j in range(len(i)):
            if i[j] == 3:
                if j != 0 :
                    if i[j-1] in [2,3]:
                        end = start+1
                        region.append((start,end))
                        start = end
                        num_pre += 1
                    else:
                        end = j
                        region.append((start,end))
                        start = end
                        end = start+1
                        region.append((start,end))
                        start = end
                        num_pre += 2
                else:
                    end = start+1
                    region.append((start,end))
                    start = end
                    num_pre += 1

            elif i[j] == 2:
                end = j+1
                region.append((start,end))
                start = end
                num_pre += 1
            elif i[j] == 1:
                if j!= 0 :
                    if i[j-1] not in [0,1]:
                        start = j
            else:
                if j != 0 :
                    if i[j-1] not in [2,3]:
                        end = j
                        region.append((start,end))
                        start = end
                        num_pre += 1
                    else:
                        start = j
        res.append(set(region))

    num = 0
    for i,j in zip(answer,res):
        num += len(i & j)
    pre = num /  num_pre
    recall = num / num_word
    f1_score =  2*pre*recall / (pre + recall)
    return f1_score

with open("train.ctb60.hwc.txt",'r',encoding='utf8') as f:
    a=f.read()
a=a.split('\n')
b=a[:]
text_all=[]
tmp=[]
label_all=[]
text = []
label = []
for i in range(len(a)):
    a[i]=list(a[i])

for i in range(len(a)):
    tmp=[]
    for j in a[i]:
        if j != ' ':
            tmp.append(j)   
    text_all.append(tmp)

for i in range(len(text_all)):
    text_all[i] = ''.join(text_all[i])


for i in range(len(b)):
    b[i]=b[i].split()


for i in range(len(b)):
    str=''
    for j in range(len(b[i])):
        if len(b[i][j])== 1:
            b[i][j]="O"
            str+=b[i][j]

 
        elif len(b[i][j])== 2:
            b[i][j]='BE'
            str+=b[i][j]
        else:
            b[i][j]="B"+"I"*(len(b[i][j])-2)+"E"

            str+=b[i][j]
    label_all.append(list(str))

label_idx ={"B":0,"I":1,"E":2,"O":3}

for i in range(len(label_all)):
    for j in range(len(label_all[i])):
        label_all[i][j] = label_idx[label_all[i][j]]
for i in range(len(label_all)):
    label_all[i] = torch.tensor(label_all[i])



for i in range(len(text_all)):
    if len(text_all[i]) <= 375:
        text.append(text_all[i])
        label.append(label_all[i])


for i in range(len(text)):
    tmp = list(text[i])
    tmp2 = []
    for j in range(len(tmp)):
        x = ord(tmp[j])
        if  (x>47 and x<58) or (x > 64 and x < 91) or (x>96 and x<123) or  tmp[j] == '○'  or tmp[j] == '─':
            tmp2.append(tmp[j])
            tmp2.append(" ")
        else:
            tmp2.append(tmp[j])
 
    text[i] = ''.join(tmp2)

del text[5222]
del label[5222]



assert len(text) == len(label),"长度不同"

#label = pad_sequence(label, padding_value=0,batch_first=True)


with open("test.ctb60.hwc.txt",'r',encoding='utf8') as f:
    a=f.read()
a=a.split('\n')
b=a[:]
test_text_all=[]
tmp=[]
answer_all=[]
test_text = []
answer = []

for i in range(len(a)):
    a[i]=list(a[i])

for i in range(len(a)):
    tmp=[]
    for j in a[i]:
        if j != ' ':
            tmp.append(j)   
    test_text_all.append(tmp)

for i in range(len(test_text_all)):
    test_text_all[i] = ''.join(test_text_all[i])

for i in range(len(b)):
    b[i]=b[i].split()


for i in range(len(b)):
    region = []
    start = 0
    for word in b[i]:
        end = start + len(word)
        region.append((start, end))
        start = end
    answer_all.append(set(region))


for i in range(len(test_text_all)):
    if len(test_text_all[i]) <= 375:
        test_text.append(test_text_all[i])
        answer.append(answer_all[i])
for i in range(len(test_text)):
    tmp = list(test_text[i])
    tmp2 = []
    for j in range(len(tmp)):
        x = ord(tmp[j])
        if  (x>47 and x<58) or (x > 64 and x < 91) or (x>96 and x<123) or  tmp[j] == '○'  or tmp[j] == '─':
            tmp2.append(tmp[j])
            tmp2.append(" ")
        else:
            tmp2.append(tmp[j])
    test_text[i] = ''.join(tmp2)



toke = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')




with open("/home/hsl/hslcode/train.bilingual.txt",'r',encoding='utf8') as f:
    a=f.read()
a=a.split('\n')
fenci_train_text = []
fenci_train_text_eng = []
fenci_train_text_f = []
fenci_train_text_eng_f = []

for i in range(len(a)):
    fenci_train_text.append(a[i].split("|||||")[0])
    fenci_train_text_eng.append(a[i].split("|||||")[1])

for i in range(len(fenci_train_text)):
    if "..." in  fenci_train_text[i]:
       fenci_train_text[i]  =  fenci_train_text[i].replace("...","…")
       fenci_train_text_eng[i] = fenci_train_text_eng[i].replace("...","…")
    if len(fenci_train_text[i]) <= 375:
        fenci_train_text_f.append(fenci_train_text[i])
        fenci_train_text_eng_f.append(fenci_train_text_eng[i])

for i in range(len(fenci_train_text_f)):
    tmp = list(fenci_train_text_f[i])
    tmp2 = []
    for j in range(len(tmp)):
        x = ord(tmp[j])
        if  (x>47 and x<58) or (x > 64 and x < 91) or (x>96 and x<123) or  tmp[j] == '○'  or tmp[j] == '─':
            tmp2.append(tmp[j])
            tmp2.append(" ")
        else:
            tmp2.append(tmp[j])
 
    fenci_train_text_f[i] = ''.join(tmp2)

del fenci_train_text_f[5222]
del fenci_train_text_eng_f[5222]

train_text_multiligual = []

for i in range(len(fenci_train_text_f)):
    tmp_all = []
    for j in range(len(fenci_train_text_f[i])):
        if fenci_train_text_f[i][j]!=" ":
            tmp = fenci_train_text_f[i][:j] + " ¶ " +fenci_train_text_f[i][j]+ " ¶ " +fenci_train_text_f[i][j+1:]
            tmp_all.append((fenci_train_text_eng_f[i],tmp))
    train_text_multiligual.append(tmp_all)


with open("/home/hsl/hslcode/test.bilingual.txt",'r',encoding='utf8') as f:
    a=f.read()
a=a.split('\n')
fenci_test_text = []
fenci_test_text_eng = []
fenci_test_text_f = []
fenci_test_text_eng_f = []

for i in range(len(a)):
    fenci_test_text.append(a[i].split("|||||")[0])
    fenci_test_text_eng.append(a[i].split("|||||")[1])

for i in range(len(fenci_test_text)):
    if "..." in  fenci_test_text[i]:
       fenci_test_text[i]  =  fenci_test_text[i].replace("...","…")
       fenci_test_text_eng[i] = fenci_test_text_eng[i].replace("...","…")
    if len(fenci_test_text[i]) <= 375:
        fenci_test_text_f.append(fenci_test_text[i])
        fenci_test_text_eng_f.append(fenci_test_text_eng[i])

for i in range(len(fenci_test_text_f)):
    tmp = list(fenci_test_text_f[i])
    tmp2 = []
    for j in range(len(tmp)):
        x = ord(tmp[j])
        if  (x>47 and x<58) or (x > 64 and x < 91) or (x>96 and x<123) or  tmp[j] == '○'  or tmp[j] == '─':
            tmp2.append(tmp[j])
            tmp2.append(" ")
        else:
            tmp2.append(tmp[j])
 
    fenci_test_text_f[i] = ''.join(tmp2)


test_text_multiligual = []

for i in range(len(fenci_test_text_f)):
    tmp_all = []
    for j in range(len(fenci_test_text_f[i])):
        if fenci_test_text_f[i][j]!=" ":
            tmp = fenci_test_text_f[i][:j] + " ¶ " +fenci_test_text_f[i][j]+ " ¶ " +fenci_test_text_f[i][j+1:]
            tmp_all.append((fenci_test_text_eng_f[i],tmp))
    test_text_multiligual.append(tmp_all)




class bertdataset(Dataset):
    def __init__(self,text,label,fenci_text):
        super(bertdataset,self).__init__()
        self.text = text
        self.label = label
        '''
        input = self.tokenizer(
            self.text,
            truncation=True,
            padding = True,
            return_tensors = "pt",
            add_special_tokens= False
        )
        self.ids = input["input_ids"]
        self.mask = input["attention_mask"]
        self.token_type_ids = input["token_type_ids"]
        assert self.ids.size() == self.mask.size() ,"大小不同"
        assert fenci_start.size() == self.ids.size() , "起始数据大小不一致"
        assert fenci_end.size() == self.ids.size() , "终止数据大小不一致"
        '''
        self.fenci_text = fenci_text
    def __len__(self):
        return len(self.text)
    def __getitem__(self,idx):
        return self.text[idx],self.label[idx],self.fenci_text[idx]

class testdataset(Dataset):
    def __init__(self,text,fenci_text):
        super(testdataset,self).__init__()
        self.text = text
        self.fenci_text = fenci_text

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,idx):
        return  self.text[idx],self.fenci_text[idx]

def collate_fn_train(batch):
    a = []
    b = []
    c = []
    fenci_len = []
    questions = []
    answers = []
    for text,label,fenci_text in batch:
        a.append(text)
        b.append(label)
        c.append(fenci_text)
    
    input = toke(
                    a,
            truncation=True,
            padding = True,
            return_tensors = "pt",
            add_special_tokens= False
    )
    ids = input["input_ids"]
    mask = input["attention_mask"]
    token_type_ids = input["token_type_ids"]
    
    b = pad_sequence(b, padding_value=0,batch_first=True)

    
    for i in range(len(c)):
        fenci_len.append(len(c[i]))
        for j in range(len(c[i])):
            questions.append(c[i][j][0])
            answers.append(c[i][j][1])
    assert ids.size(1) == max(fenci_len),f"{text}"
    encodings = tokenizer(questions,answers, truncation=True,padding=True)

    return {key: torch.tensor(val) for key, val in encodings.items()},ids,mask,token_type_ids,b,fenci_len 


def collate_fn_test(batch):
    a = []
    c = []
    questions = []
    answers = []
    fenci_len = []

    for text,fenci_text in batch:
        a.append(text)
        c.append(fenci_text)

    input = toke(
                    a,
            truncation=True,
            padding = True,
            return_tensors = "pt",
            add_special_tokens= False
    )
    ids = input["input_ids"]
    mask = input["attention_mask"]
    token_type_ids = input["token_type_ids"]
    for i in range(len(c)):
        fenci_len.append(len(c[i]))
        for j in range(len(c[i])):
            questions.append(c[i][j][0])
            answers.append(c[i][j][1])
    encodings = tokenizer(questions,answers, truncation=True,padding=True)
    test_len = torch.sum(mask,dim = 1)

    return {key: torch.tensor(val) for key, val in encodings.items()},ids,mask,token_type_ids,test_len,fenci_len 



model = bishemodel.bertmodel()
model.to(device)

'''
param_optimizer = list(model.named_parameters())
for i in  param_optimizer:
    print(i[0])
'''

'''
param_optimizer = list(model.named_parameters())
#print(param_optimizer)
no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
optimizer_parameters = [
    {"params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],"weight_decay":0.001},
    {"params": [p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],"weight_decay":0.0}]
'''
optimizer = AdamW([{'params':model.bert.parameters()},{'params':model.linear.parameters(),'lr':3e-3}],lr=3e-5)
dataset = bertdataset(text,label,train_text_multiligual)
dataloader = DataLoader(dataset,batch_size= 10,collate_fn = collate_fn_train)

test_dataset = testdataset(test_text,test_text_multiligual)
test_dataloader = DataLoader(test_dataset,batch_size= 10,collate_fn = collate_fn_test)


'''
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps= int(10*len(text)/10)#此处为总轮数
)
'''
#test_ids = test_ids.to(device)
#test_mask = test_mask.to(device)
#test_type = test_type.to(device)



print("准备完毕")

best_score = 0





for num in range(10):
    model.train()
    for bi,(batch,ids,mask,token_type_ids,label,fenci_len) in tqdm(enumerate(dataloader),total= len(dataloader)):
        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = 8
            begin = 0
            result_train_start = torch.Tensor().to(device)
            result_train_end = torch.Tensor().to(device)
            for epoch in range((input_ids.size(0)//batch_size)+1):
                fin = begin+batch_size
                outputs = fenci_model(input_ids[begin:fin], attention_mask=attention_mask[begin:fin],output_hidden_states=True)
                start = torch.argmax(outputs.start_logits,dim = 1).cpu().detach().numpy().tolist()
                end = torch.argmax(outputs.end_logits,dim = 1).cpu().detach().numpy().tolist()
                #result_s = torch.stack([outputs.hidden_states[12][j,start[j],:] for j in range(len(start))],dim = 0)
                #result_e = torch.stack([outputs.hidden_states[12][j,end[j],:] for j in range(len(end))],dim = 0)
                result_s = outputs.hidden_states[12][torch.arange(len(start)),start]
                result_e = outputs.hidden_states[12][torch.arange(len(end)),end]
                result_train_start = torch.cat([result_train_start,result_s],dim = 0)
                result_train_end = torch.cat([result_train_end,result_e],dim = 0)
                begin = fin
                if begin == input_ids.size(0):
                    break
        start = 0
        end = 0
        train_start = []
        train_end = []
        for i in range(len(fenci_len)): 
            len_i = fenci_len[i]
            end = start+len_i
            train_start_i = result_train_start[start:end]
            train_end_i = result_train_end[start:end]
            train_start.append(train_start_i)
            train_end.append(train_end_i)
            start = end
        train_start = pad_sequence(train_start, padding_value=0,batch_first=True)
        train_end = pad_sequence(train_end, padding_value=0,batch_first=True)

        optimizer.zero_grad()
        loss= model(
            ids,
            mask,
            token_type_ids,
            label,
            device,
            train_start,
            train_end
        )
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #print(f"轮数{i},次数{bi}:{loss}")
    print(f"轮数{num}训练完成"+"--"*10)
    model.eval()
    result = []
    for bi,(batch,ids,mask,token_type_ids,test_len,fenci_len) in tqdm(enumerate(test_dataloader),total= len(test_dataloader)):

        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = 8
            begin = 0
            result_test_start = torch.Tensor().to(device)
            result_test_end = torch.Tensor().to(device)
            for epoch in range((input_ids.size(0)//batch_size)+1):
                fin = begin+batch_size
                outputs = fenci_model(input_ids[begin:fin], attention_mask=attention_mask[begin:fin],output_hidden_states=True)
                start = torch.argmax(outputs.start_logits,dim = 1).cpu().detach().numpy().tolist()
                end = torch.argmax(outputs.end_logits,dim = 1).cpu().detach().numpy().tolist()
                result_s = torch.stack([outputs.hidden_states[12][j,start[j],:] for j in range(len(start))],dim = 0)
                result_e = torch.stack([outputs.hidden_states[12][j,end[j],:] for j in range(len(end))],dim = 0)
                result_test_start = torch.cat([result_test_start,result_s],dim = 0)
                result_test_end = torch.cat([result_test_end,result_e],dim = 0)
                begin = fin 
                if begin == input_ids.size(0):
                    break

        start = 0
        end = 0
        test_start = []
        test_end = []
        for i in range(len(fenci_len)): 
            len_i = fenci_len[i]
            end = start+len_i
            test_start_i = result_test_start[start:end]
            test_end_i = result_test_end[start:end]
            test_start.append(test_start_i)
            test_end.append(test_end_i)
            start = end

        test_start = pad_sequence(test_start, padding_value=0,batch_first=True)
        test_end = pad_sequence(test_end, padding_value=0,batch_first=True)
           
   

        result_tmp = model.get_score(
            ids,
            mask,
            token_type_ids,
            test_start,
            test_end
        )

        result_tmp = result_tmp.cpu().detach().numpy().tolist()

        for i in range(len(ids)):
            result_tmp[i] = result_tmp[i][:test_len[i]]
        result.extend(result_tmp)
    assert len(result) == len(answer),"长度不一致"
    f1_score = f1(answer,result)
    print(f"轮数{num}f1分数{f1_score}")
    if f1_score > best_score:
        torch.save(model,"/home/hsl/hslcode/bishe_model_fenci2.pth")
        best_score = f1_score

print(f"最好分数{best_score}")
