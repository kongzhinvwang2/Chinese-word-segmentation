import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import qinggan2
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer,BertModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
    if len(text_all[i]) <= 300:
        text.append(text_all[i])
        label.append(label_all[i])

assert len(text) == len(label),"长度不同"

label = pad_sequence(label, padding_value=0,batch_first=True)



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
    if len(test_text_all[i]) <= 300:
        test_text.append(test_text_all[i])
        answer.append(answer_all[i])




toke = BertTokenizer.from_pretrained("bert-base-chinese")
test_in = toke(
    test_text,
    truncation=True,
    padding = True,
    return_tensors= "pt",
    add_special_tokens=False
)


test_ids = test_in["input_ids"]
test_mask = test_in["attention_mask"]
test_type = test_in["token_type_ids"]

test_len = torch.sum(test_mask,dim = 1)


class bertdataset(Dataset):
    def __init__(self,text,label):
        super(bertdataset,self).__init__()
        self.text = text
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        input = self.tokenizer(
            self.text,
            truncation=True,
            padding = True,
            return_tensors = "pt",
            add_special_tokens=False
        )
        self.ids = input["input_ids"]
        self.mask = input["attention_mask"]
        self.token_type_ids = input["token_type_ids"]
        assert self.ids.size() == self.mask.size() ,"大小不同"

    def __len__(self):
        return len(self.text)
    def __getitem__(self,item):
        return self.ids[item],self.mask[item],self.token_type_ids[item],self.label[item]

class testdataset(Dataset):
    def __init__(self,ids,mask,type,test_len):
        super(testdataset,self).__init__()
        self.ids = ids
        self.mask = mask
        self.type = type
        self.test_len = test_len
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,item):
        return  self.ids[item],self.mask[item],self.type[item],self.test_len[item]




device = torch.device("cuda")
model = qinggan2.bertmodel()
model.to(device)

param_optimizer = list(model.named_parameters())
for i in  param_optimizer:
    print(i[0])


'''
param_optimizer = list(model.named_parameters())
#print(param_optimizer)
no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
optimizer_parameters = [
    {"params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],"weight_decay":0.001},
    {"params": [p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],"weight_decay":0.0}]
'''
optimizer = AdamW(model.parameters(),lr=3e-5)
dataset = bertdataset(text,label)
dataloader = DataLoader(dataset,batch_size= 10)

test_dataset = testdataset(test_ids,test_mask,test_type,test_len)
test_dataloader = DataLoader(test_dataset,batch_size= 10)



scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps= int(10*len(text)/10)#此处为总轮数
)

#test_ids = test_ids.to(device)
#test_mask = test_mask.to(device)
#test_type = test_type.to(device)



print("准备完毕")

best_score = 0


'''

for i in range(10):
    model.train()
    for bi,(ids,mask,token_type_ids,label) in tqdm(enumerate(dataloader),total= len(dataloader)):
        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss= model(
            ids,
            mask,
            token_type_ids,
            label,
            device
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"轮数{i},次数{bi}:{loss}")
    print(f"轮数{i}训练完成"+"--"*10)
    model.eval()
    result = []
    for bi,(ids,mask,token_type_ids,test_len) in tqdm(enumerate(test_dataloader),total= len(test_dataloader)):

        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)

        result_tmp = model.get_score(
            ids,
            mask,
            token_type_ids
        )

        result_tmp = result_tmp.cpu().detach().numpy().tolist()

        for i in range(len(ids)):
            result_tmp[i] = result_tmp[i][:test_len[i]]
        result.extend(result_tmp)
    assert len(result) == len(answer),"长度不一致"
    f1_score = f1(answer,result)
    if f1_score > best_score:
        torch.save(model,"/home/hsl/hslcode/bert_model.pth")
        best_score = f1_score

print(f"最好分数{best_score}")
'''



    

'''
#print(optimizer_parameters)
accuracy = 1 #预测准确率
best_accuracy = 0
for epoch in range(config.EPOCH):
    if accuracy > best_accuracy:
        torch.save(model,"/home/hsl/hslcode/bert_model.pth")
'''
