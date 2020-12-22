import torch
from torch.autograd import Variable
import json
from torch_train import LogisticRegression
import math
import argparse


BATCH_SIZE=8

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--test_file', type=str, default='./test_temp.json')
    parser.add_argument('-m', '--model_file', type=str, default='./model.pkl')

    args=parser.parse_args()
    test_file=args.test_file
    model_file=args.model_file

    model=torch.load(model_file)
    model.eval()

    with open(test_file) as f:
        res=f.read()
        test_set=json.loads(res)

    x=[]
    y=[]
    for item in test_set:
        item["data"]=item["data"].split(" ")
        word_list=item["data"]
        for i in range(len(word_list)):
            if word_list[i] in model.vocab:
                word_list[i]=model.vocab[word_list[i]]
            else:
                word_list[i]=1
        x.append(word_list)
        y.append(item["label"])

    x_batches=[]
    y_batches=[]
    num_of_batches=math.ceil(len(x)/BATCH_SIZE)
    for i in range(num_of_batches):
        j=i*BATCH_SIZE
        x_batches.append(x[j:j+BATCH_SIZE])
        y_batches.append(y[j:j+BATCH_SIZE])

    for batch_x in x_batches:
        maxlength=0
        for sentence in batch_x:
            if len(sentence)>maxlength:
                maxlength=len(sentence)
        for sentence in batch_x:
            if len(sentence)<maxlength:
                for i in range(maxlength-len(sentence)):
                    sentence.append(0)

    acc=0
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
        mask=out.ge(0.5).int()

        correct=(mask==batch_y).sum()
        acc+=correct.item()

    print('test set acc = {:.9f}'.format(acc/len(test_set)))