import torch
from torch.autograd import Variable
import json
from torch_train import LogisticRegression
import math
import argparse

BATCH_SIZE=8

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--test_file', type=str, default='./testdataexample')
    parser.add_argument('-m', '--model_file', type=str, default='./model.pkl')

    args=parser.parse_args()
    test_file=args.test_file
    model_file=args.model_file

    model=torch.load(model_file)
    model.eval()

    with open(test_file) as f:
        res=f.read()
        test_set=json.loads(res)

    # for i in range(3):
    #     test_set+=test_set
    
    for i in range(len(test_set)):
        test_set[i]=test_set[i].split(" ")
        for j in range(len(test_set[i])):
            if test_set[i][j] in model.vocab:
                test_set[i][j]=model.vocab[test_set[i][j]]
            else:
                test_set[i][j]=1

    x_batches=[]
    num_of_batches=math.ceil(len(test_set)/BATCH_SIZE)
    for i in range(num_of_batches):
        j=i*BATCH_SIZE
        x_batches.append(test_set[j:j+BATCH_SIZE])

    for batch_x in x_batches:
        maxlength=0
        for sentence in batch_x:
            if len(sentence)>maxlength:
                maxlength=len(sentence)
        for sentence in batch_x:
            if len(sentence)<maxlength:
                for i in range(maxlength-len(sentence)):
                    sentence.append(0)

    f=open("./output.txt", "a")
    f.seek(0)
    f.truncate()

    for batch_x in x_batches:
        batch_x=torch.LongTensor(batch_x)
        if torch.cuda.is_available():
            batch_x=Variable(batch_x).cuda()
        else:
            batch_x=Variable(batch_x)

        out=model(batch_x).squeeze(1)
        mask=out.ge(0.5).int()

        res=torch.Tensor.cpu(mask)
        res=res.numpy()

        for pred in res:
            f.write(str(pred)+"\n")

    f.flush()
    f.close()