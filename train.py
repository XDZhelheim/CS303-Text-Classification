from torch import nn, optim
from torch.autograd import Variable
import torch
import json
import math
import argparse
import random

BATCH_SIZE=32
EMB_DIM=128
# HIDDEN_DIM=128

class LogisticRegression(nn.Module):
    def __init__(self, vocab):
        super(LogisticRegression, self).__init__()
        self.vocab=vocab
        self.emb=nn.Embedding(len(vocab), EMB_DIM, 0)
        # self.lstm=nn.LSTM(EMB_DIM, HIDDEN_DIM)
        self.lr=nn.Linear(EMB_DIM, 1) 
        self.sig=nn.Sigmoid() 

    def fold(self, x):
        return torch.sum(x, dim=1)

    def forward(self, x):
        x=self.emb(x) # batch_size * seq_len * emb_dim
        # x, (_, _)=self.lstm(x) # batch_size * seq_len * hidden_dim
        x=self.fold(x) # batch_size * hidden_dim
        x=self.lr(x) # batch_size *1
        x=self.sig(x)
        return x

if __name__ == "__main__":
#---args--------------------------------------------------------------------------------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str, default='D:\\Codes\\PythonWorkspace\\AI_Lab\\LogisticRegression\\train_temp.json')

    args=parser.parse_args()
    train_file=args.train_file

#---data -------------------------------------------------------------------------------------------------
    with open(train_file) as f:
        res=f.read()
        train_set=json.loads(res)

    vocab={}
    vocab[0]="pad"
    vocab[1]="unk"

    count=2
    for item in train_set:
        item["data"]=item["data"].split(" ")
        for word in item["data"]:
            if word not in vocab:
                vocab[word]=count
                count+=1

    # m=0
    # for item in train_set:
    #     if len(item["data"])>m:
    #         m=len(item["data"])
    # print(m)

    # train_set=sorted(train_set, key=lambda i: len(i["data"]))
    random.shuffle(train_set)

    x=[]
    y=[]
    for item in train_set:
        word_list=item["data"]
        for i in range(len(word_list)):
            word_list[i]=vocab[word_list[i]]
        x.append(word_list)
        y.append(item["label"])

#---batch-------------------------------------------------------------------------------------------------
    x_batches=[]
    y_batches=[]
    num_of_batches=math.ceil(len(x)/BATCH_SIZE)
    for i in range(num_of_batches):
        j=i*BATCH_SIZE
        x_batches.append(x[j:j+BATCH_SIZE])
        y_batches.append(y[j:j+BATCH_SIZE])
    
#---to fixed length matrix -------------------------------------------------------------------------------
    for batch_x in x_batches:
        maxlength=0
        for sentence in batch_x:
            if len(sentence)>maxlength:
                maxlength=len(sentence)
        for sentence in batch_x:
            if len(sentence)<maxlength:
                for i in range(maxlength-len(sentence)):
                    sentence.append(0)

    # batch_x: num_of_batches * batch_size * seq_len = 3125 * 8 * maxlength

    # for sentence in x_batches[0]:
    #     print(len(sentence))

#---train -------------------------------------------------------------------------------------------------
    model=LogisticRegression(vocab)
    if torch.cuda.is_available():
        model.cuda()

    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(), lr=1e-3)

    epochs=25
    for epoch in range(epochs):
        print_loss=0
        c=0
        for i in range(num_of_batches):
            batch_x=torch.LongTensor(x_batches[i])
            batch_y=torch.FloatTensor(y_batches[i])

            if torch.cuda.is_available():
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()
            else:
                batch_x=Variable(batch_x)
                batch_y=Variable(batch_y)
            
            out=model(batch_x).squeeze(1)
            mask=out.ge(0.5).int()  # >= 0.5 -> 1

            loss=criterion(out, batch_y)
            # loss=torch.mean(torch.clamp(1-batch_y*out, min=0)) # SVM
            print_loss+=loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # update grad

            correct=(mask==batch_y).sum()
            c+=correct.item()

        acc=c/len(train_set)

        print('epoch {}'.format(epoch+1))
        print('loss = {:.9f}'.format(print_loss/num_of_batches))
        print('acc = {:.9f}'.format(acc))
        print("-------------------------------")

        if acc>=0.9999:
            break

#---save model parameters-----------------------------------------------------------------------
    torch.save(model, "./model.pkl")
