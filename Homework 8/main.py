from __future__ import annotations
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class myNaiveBayes:

    def fit(self, X:pd.Series, y:pd.Series) -> myNaiveBayes:
        import regex as re

        y_index_groups = y.groupby(y).apply(lambda g: g.index.tolist())
        self.label_probs = y_index_groups.apply(lambda l: len(l))
        self.label_probs = self.label_probs / sum(self.label_probs)

        self.condition_probs = {}
        vocabs:set = set()
        for label in y_index_groups.index.to_list():
            self.condition_probs[label] = pd.Series()
            for doc in X[y_index_groups[label]]:
                for word in re.split(r"\W+", doc):
                    vocabs.add(word)
                    self.condition_probs[label].loc[word] = self.condition_probs[label].get(word, 0) + 1
        self.default_probs = {}
        for label, prob in self.condition_probs.items():
            self.condition_probs[label] = (prob + 1) / (sum(prob) + len(vocabs))
            self.default_probs[label] = 1 / (sum(prob) + len(vocabs))
        return self

    def predict(self, X:pd.Series) -> pd.Series:
        import regex as re

        result = None
        for doc in X:
            doc_probs = np.log10(self.label_probs)
            for word in re.split(r"\W+", doc):
                word_probs = pd.Series()
                for label, prob in self.condition_probs.items():
                    word_probs.loc[label] = np.log10(prob.get(word, self.default_probs[label]))
                doc_probs += word_probs
            if result is None:
                result = pd.Series([doc_probs.idxmax()])
            else:
                result = pd.concat([result, pd.Series([doc_probs.idxmax()])], ignore_index=True)
        return result

class myTfIdf:
    def fit(self, X:pd.Series, y:pd.Series=None) -> myTfIdf:
        import regex as re
        
        self.vocabs = set()
        for doc in X:
            self.vocabs.update(word for word in re.split(r"\W+", doc) if word)
        
        tf = pd.DataFrame(columns=list(self.vocabs))
        for doc in X:
            tf.loc[len(tf)] = 0
            for word in re.split(r"\W+", doc):
                if not word:
                    continue
                tf.loc[tf.index[-1], word] += 1
        idf_temp = {}
        for vocab in self.vocabs:
            idf_temp[vocab] = [np.log10(len(X) / tf[vocab].gt(0).sum())]
        self.idf = pd.DataFrame(idf_temp, columns=list(self.vocabs))
        return self

    def transform(self, X:pd.Series) -> pd.DataFrame:
        import regex as re
        tf = pd.DataFrame(columns=list(self.vocabs))
        for doc in X:
            tf.loc[len(tf)] = 0
            for word in re.split(r"\W+", doc):
                if not word or word not in self.vocabs:
                    continue
                tf.loc[tf.index[-1], word] += 1
        _tfidf = tf.multiply(self.idf.iloc[0], axis='columns')
        return _tfidf

class Preprocessor:
    def feature(self, txt:str) -> str:
        return txt.lower()
    
    def label(self, txt:str) -> int:
        if txt == "ham":
            return 0
        return 1


def Q2(df:pd.DataFrame) -> None:

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, precision_score, recall_score

    df_train, df_test = train_test_split(df)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    myNB = myNaiveBayes().fit(df_train["Message"], df_train["Category"])
    y = myNB.predict(df_test["Message"])
    print(f1_score(df_test["Category"], y))
    print(precision_score(df_test["Category"], y))
    print(recall_score(df_test["Category"], y))

def eval_clf(clf:BaseEstimator, grid:dict, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray) -> None:
    
    from textwrap import dedent
    from matplotlib import pyplot as plt
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import GridSearchCV
    
    scorer = make_scorer(f1_score)

    clf_cv = GridSearchCV(clf, grid, scoring=scorer, cv=5, n_jobs=-1, verbose=10)
    clf_cv.fit(train_x, train_y)
    best_clf = clf_cv.best_estimator_

    guess_y = best_clf.predict(test_x)

    clf_name = type(clf).__name__
    with open("./build/{}.txt".format(clf_name), "w") as reportfile:
        reportfile.writelines("{}\n".format(clf_name))
        reportfile.writelines(dedent("""
                              < Best Parameters >
                              """
        ))
        for name, value in clf_cv.best_params_.items():
            reportfile.writelines("{}: {}\n".format(name, value))
        reportfile.writelines(dedent("""
                           < Score >
                           F1: {}
                           Precision: {}
                           Recall: {}]\n
                           """.format(f1_score(test_y, guess_y),
                                      precision_score(test_y, guess_y),
                                      recall_score(test_y, guess_y))
        ))

    cm = confusion_matrix(test_y, guess_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    _, ax = plt.subplots(figsize=(15,15))
    disp.plot(ax=ax,cmap=plt.cm.Blues)
    disp.ax_.set_title(clf_name)
    plt.savefig("./build/{}.png".format(clf_name), bbox_inches='tight')
    plt.clf()

def Q3(df:pd.DataFrame) -> None:

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(df)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    myTI = myTfIdf().fit(df_train["Message"])
    train_x = myTI.transform(df_train["Message"]).to_numpy()
    train_y = df_train["Category"].to_numpy()
    test_x = myTI.transform(df_test["Message"]).to_numpy()
    test_y = df_test["Category"].to_numpy()

    clf = LogisticRegression(penalty='l2', random_state=0, max_iter=200)
    grid = {
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
    eval_clf(clf, grid, train_x, train_y, test_x, test_y)

def Q4(df:pd.DataFrame) -> None:

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    df_train, df_test = train_test_split(df)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    vct = CountVectorizer().fit(df_train["Message"])
    train_x = vct.transform(df_train["Message"])
    train_y = df_train["Category"].to_numpy()
    test_x = vct.transform(df_test["Message"])
    test_y = df_test["Category"].to_numpy()

    clf = MultinomialNB()
    grid = {
                "alpha": [50, 15, 10, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01,  0.001],
                "fit_prior": [True, False]
            }
    eval_clf(clf, grid, train_x, train_y, test_x, test_y)

    vct = TfidfVectorizer(norm=None).fit(df_train["Message"])
    train_x = vct.transform(df_train["Message"])
    test_x = vct.transform(df_test["Message"])
    clf = LogisticRegression(penalty='l2', random_state=0, max_iter=200)
    grid = {
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
    eval_clf(clf, grid, train_x, train_y, test_x, test_y)

class myDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=510, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label)}

def Q5(df:pd.DataFrame) -> None:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer, BertForSequenceClassification
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split, StratifiedKFold

    df_train, df_test = train_test_split(df)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    myDS = myDataset(df_train["Message"].tolist(), df_train["Category"].tolist(), tokenizer)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (train_indicies, valid_indicies) in enumerate(skf.split(df_train["Message"], df_train["Category"])):
        loader_train = DataLoader(Subset(myDS, train_indicies), batch_size=32, shuffle=True)
        loader_valid = DataLoader(Subset(myDS, valid_indicies), batch_size=32, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # 1e-5, 2e-5, 5e-5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        model.train()
        for epoch in range(20):
            for batch in loader_train:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in loader_valid:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted_labels = torch.max(outputs.logits, dim=1)
                val_predictions.extend(predicted_labels.tolist())
                val_labels.extend(labels.tolist())
    
    myDS = myDataset(df_test["Message"].tolist(), df_test["Category"].tolist(), tokenizer)
    loader_test = DataLoader(myDS, batch_size=32, shuffle=False)

    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for batch in loader_test:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            test_predictions.extend(predicted_labels.tolist())
            test_labels.extend(labels.tolist())
    print(f1_score(test_labels, test_predictions))
    print(precision_score(test_labels, test_predictions))
    print(recall_score(test_labels, test_predictions))

def main():
    pp = Preprocessor()
    df = pd.read_csv("./data/data.csv")
    df["Message"] = df["Message"].apply(pp.feature)
    df["Category"] = df["Category"].apply(pp.label)
    Q2(df)
    
 
if __name__ == "__main__":
    main()