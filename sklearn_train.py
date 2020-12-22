from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import json
import argparse
import time

class TextClassification():
    def __init__(self):
        self.text_clf_svm = Pipeline([('vect', CountVectorizer()), 
        ('tfidf', TfidfTransformer()), 
        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))
        ])

        self.parameters_svm={'vect__ngram_range': [(1, 2), (1, 3)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-4, 1e-5)}

        self.gs_clf_svm = GridSearchCV(self.text_clf_svm, self.parameters_svm, n_jobs=-1)

if __name__ == "__main__":
#---args--------------------------------------------------------------------------------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str, default='./train.json')

    args=parser.parse_args()
    train_file=args.train_file

#---data -------------------------------------------------------------------------------------------------
    with open(train_file) as f:
        res=f.read()
        train_set=json.loads(res)

    x=[]
    y=[]
    for item in train_set:
        x.append(item["data"])
        y.append(item["label"])

#---train -------------------------------------------------------------------------------------------------
    start=time.time()

    model=TextClassification()
    model.gs_clf_svm.fit(x, y)

    end=time.time()

    print("param:", model.gs_clf_svm.best_params_)
    print("score:", model.gs_clf_svm.best_score_)

#---save model-----------------------------------------------------------------------
    joblib.dump(model, "./model.joblib")

    print("Train Finished")
    print("Time:", end-start)