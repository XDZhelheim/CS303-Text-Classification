from sklearn_train import TextClassification
import argparse
import json
import joblib
import numpy as np

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--test_file', type=str, default='./test_temp.json')
    parser.add_argument('-m', '--model_file', type=str, default='./model.joblib')

    args=parser.parse_args()
    test_file=args.test_file
    model_file=args.model_file

    with open(test_file) as f:
        res=f.read()
        test_set=json.loads(res)

    x=[]
    y=[]
    for item in test_set:
        x.append(item["data"])
        y.append(item["label"])

    model=joblib.load(model_file)

    y_pred=model.gs_clf_svm.predict(x)

    print("acc =", np.mean(y_pred==y))