from sklearn_train import TextClassification
import argparse
import json
import joblib

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--test_file', type=str, default='./testdataexample')
    parser.add_argument('-m', '--model_file', type=str, default='./model.joblib')

    args=parser.parse_args()
    test_file=args.test_file
    model_file=args.model_file

    with open(test_file) as f:
        res=f.read()
        test_set=json.loads(res)

    model=joblib.load(model_file)

    y_pred=model.gs_clf_svm.predict(test_set)

    f=open("./output.txt", "a")
    f.seek(0)
    f.truncate()

    for pred in y_pred:
        f.write(str(pred)+"\n")

    f.flush()
    f.close()