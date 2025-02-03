from __future__ import annotations

import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

class FAIBaseModel:
    def __init__(self) -> None:
        pass

    def fit(self, X:pd.Series, y:pd.Series) -> FAIBaseModel:
        print("{} <Training> {}".format(datetime.now(), type(self).__name__))
    
    def predict(self, X:pd.Series) -> pd.Series:
        pass
    
    def evaluate(self, X:pd.Series, y:pd.Series) -> None:
        print("{} <Evaluation> {}".format(datetime.now(), type(self).__name__))


class FAINaiveBase(FAIBaseModel):
    from sklearn.naive_bayes import MultinomialNB

    def __init__(self, cv:int=5) -> None:
        super().__init__()
        self.cv = cv
    
    def fit(self, X:pd.Series, y:pd.Series) -> FAINaiveBase:
        super().fit(X, y)

        grid = {"alpha": [50, 15, 10, 5, 1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01,  0.001],
                "fit_prior": [True, False]}
        cv = GridSearchCV(FAINaiveBase.MultinomialNB(),
                          grid,
                          scoring=make_scorer(f1_score),
                          cv=self.cv,
                          n_jobs=-1,
                          verbose=10)
        self.vectorizer = TfidfVectorizer().fit(X)
        train_x = self.vectorizer.transform(X)
        train_y = y.to_numpy()
        cv:GridSearchCV = cv.fit(train_x, train_y)
        self.best_estimator_ = cv.best_estimator_
        
        return self

    def predict(self, X:pd.Series) -> pd.Series:
        test_x = self.vectorizer.transform(X)
        return pd.Series(self.best_estimator_.predict(test_x))

    def evaluate(self, X:pd.Series, y:pd.Series) -> None:
        super().evaluate(X, y)

        from textwrap import dedent
        from matplotlib import pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

        guess_y = self.predict(X)
        file_name = type(self).__name__

        with open("./build/{}.txt".format(file_name), "w") as reportfile:
            reportfile.writelines("{}\n".format(file_name))
            reportfile.writelines(dedent("""
                                < Best Parameters >
                                """
            ))
            for name, value in self.best_estimator_.get_params().items():
                reportfile.writelines("{}: {}\n".format(name, value))
            reportfile.writelines(dedent("""
                            < Score >
                            F1: {}
                            Precision: {}
                            Recall: {}
                            """.format(f1_score(y, guess_y),
                                        precision_score(y, guess_y),
                                        recall_score(y, guess_y))
            ))
        
        _, ax = plt.subplots(figsize=(15,15))
        ConfusionMatrixDisplay(confusion_matrix(y, guess_y)).plot(ax=ax,cmap=plt.cm.Blues)
        ax.set_title(file_name)
        plt.savefig("./build/{}_confusion.png".format(file_name), bbox_inches="tight")
        plt.clf()

        fpr, tpr, _ = roc_curve(y, self.best_estimator_.predict_proba(self.vectorizer.transform(X))[:,1])
        plt.figure(figsize=(15, 15))
        plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = {:.2f})".format(auc(fpr, tpr)))
        plt.plot([0,1], [0,1], color="gray", linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} ROC Curve".format(file_name))
        plt.legend(loc="lower right")
        plt.savefig("./build/{}_roc.png".format(file_name), bbox_inches="tight")
        plt.clf()

class FAILogisticRegression(FAIBaseModel):
    from sklearn.linear_model import LogisticRegression

    def __init__(self, cv:int=5) -> None:
        super().__init__()
        self.cv = cv
    
    def fit(self, X:pd.Series, y:pd.Series) -> FAILogisticRegression:
        super().fit(X, y)

        grid = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100]}
        cv = GridSearchCV(FAILogisticRegression.LogisticRegression(),
                          grid,
                          scoring=make_scorer(f1_score),
                          cv=self.cv,
                          n_jobs=-1,
                          verbose=10)
        self.vectorizer = TfidfVectorizer().fit(X)
        train_x = self.vectorizer.transform(X)
        train_y = y.to_numpy()
        cv:GridSearchCV = cv.fit(train_x, train_y)
        self.best_estimator_ = cv.best_estimator_
        
        return self

    def predict(self, X:pd.Series) -> pd.Series:
        test_x = self.vectorizer.transform(X)
        return pd.Series(self.best_estimator_.predict(test_x))

    def evaluate(self, X:pd.Series, y:pd.Series) -> None:
        super().evaluate(X, y)

        from textwrap import dedent
        from matplotlib import pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

        guess_y = self.predict(X)
        file_name = type(self).__name__

        with open("./build/{}.txt".format(file_name), "w") as reportfile:
            reportfile.writelines("{}\n".format(file_name))
            reportfile.writelines(dedent("""
                                < Best Parameters >
                                """
            ))
            for name, value in self.best_estimator_.get_params().items():
                reportfile.writelines("{}: {}\n".format(name, value))
            reportfile.writelines(dedent("""
                            < Score >
                            F1: {}
                            Precision: {}
                            Recall: {}
                            """.format(f1_score(y, guess_y),
                                        precision_score(y, guess_y),
                                        recall_score(y, guess_y))
            ))
        
        _, ax = plt.subplots(figsize=(15,15))
        ConfusionMatrixDisplay(confusion_matrix(y, guess_y)).plot(ax=ax,cmap=plt.cm.Blues)
        ax.set_title(file_name)
        plt.savefig("./build/{}_confusion.png".format(file_name), bbox_inches="tight")
        plt.clf()

        fpr, tpr, _ = roc_curve(y, self.best_estimator_.predict_proba(self.vectorizer.transform(X))[:,1])
        plt.figure(figsize=(15, 15))
        plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = {:.2f})".format(auc(fpr, tpr)))
        plt.plot([0,1], [0,1], color="gray", linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} ROC Curve".format(file_name))
        plt.legend(loc="lower right")
        plt.savefig("./build/{}_roc.png".format(file_name), bbox_inches="tight")
        plt.clf()

class FAIAdaBoost(FAIBaseModel):
    from sklearn.ensemble import AdaBoostClassifier

    def __init__(self, cv:int=5) -> None:
        super().__init__()
        self.cv = cv
    
    def fit(self, X:pd.Series, y:pd.Series) -> FAIAdaBoost:
        super().fit(X, y)

        grid = {"n_estimators": [10, 50, 100, 500],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0]}
        cv = GridSearchCV(FAIAdaBoost.AdaBoostClassifier(),
                          grid,
                          scoring=make_scorer(f1_score),
                          cv=self.cv,
                          n_jobs=-1,
                          verbose=10)
        self.vectorizer = TfidfVectorizer().fit(X)
        train_x = self.vectorizer.transform(X)
        train_y = y.to_numpy()
        cv:GridSearchCV = cv.fit(train_x, train_y)
        self.best_estimator_ = cv.best_estimator_
        
        return self

    def predict(self, X:pd.Series) -> pd.Series:
        test_x = self.vectorizer.transform(X)
        return pd.Series(self.best_estimator_.predict(test_x))

    def evaluate(self, X:pd.Series, y:pd.Series) -> None:
        super().evaluate(X, y)

        from textwrap import dedent
        from matplotlib import pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

        guess_y = self.predict(X)
        file_name = type(self).__name__

        with open("./build/{}.txt".format(file_name), "w") as reportfile:
            reportfile.writelines("{}\n".format(file_name))
            reportfile.writelines(dedent("""
                                < Best Parameters >
                                """
            ))
            for name, value in self.best_estimator_.get_params().items():
                reportfile.writelines("{}: {}\n".format(name, value))
            reportfile.writelines(dedent("""
                            < Score >
                            F1: {}
                            Precision: {}
                            Recall: {}
                            """.format(f1_score(y, guess_y),
                                        precision_score(y, guess_y),
                                        recall_score(y, guess_y))
            ))
        
        _, ax = plt.subplots(figsize=(15,15))
        ConfusionMatrixDisplay(confusion_matrix(y, guess_y)).plot(ax=ax,cmap=plt.cm.Blues)
        ax.set_title(file_name)
        plt.savefig("./build/{}_confusion.png".format(file_name), bbox_inches="tight")
        plt.clf()

        fpr, tpr, _ = roc_curve(y, self.best_estimator_.predict_proba(self.vectorizer.transform(X))[:,1])
        plt.figure(figsize=(15, 15))
        plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = {:.2f})".format(auc(fpr, tpr)))
        plt.plot([0,1], [0,1], color="gray", linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} ROC Curve".format(file_name))
        plt.legend(loc="lower right")
        plt.savefig("./build/{}_roc.png".format(file_name), bbox_inches="tight")
        plt.clf()

class FAIXGBoost(FAIBaseModel):
    from xgboost import XGBClassifier

    def __init__(self, cv:int=5) -> None:
        super().__init__()
        self.cv = cv
    
    def fit(self, X:pd.Series, y:pd.Series) -> FAIXGBoost:
        super().fit(X, y)

        grid = {"min_child_weight": [1, 5, 10],
                "gamma": [0.5, 1, 1.5, 2, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "max_depth": [3, 4, 5]}
        cv = GridSearchCV(FAIXGBoost.XGBClassifier(),
                          grid,
                          scoring=make_scorer(f1_score),
                          cv=self.cv,
                          n_jobs=-1,
                          verbose=10)
        self.vectorizer = TfidfVectorizer().fit(X)
        train_x = self.vectorizer.transform(X)
        train_y = y.to_numpy()
        cv:GridSearchCV = cv.fit(train_x, train_y)
        self.best_estimator_ = cv.best_estimator_
        
        return self

    def predict(self, X:pd.Series) -> pd.Series:
        test_x = self.vectorizer.transform(X)
        return pd.Series(self.best_estimator_.predict(test_x))

    def evaluate(self, X:pd.Series, y:pd.Series) -> None:
        super().evaluate(X, y)

        from textwrap import dedent
        from matplotlib import pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

        guess_y = self.predict(X)
        file_name = type(self).__name__

        with open("./build/{}.txt".format(file_name), "w") as reportfile:
            reportfile.writelines("{}\n".format(file_name))
            reportfile.writelines(dedent("""
                                < Best Parameters >
                                """
            ))
            for name, value in self.best_estimator_.get_params().items():
                reportfile.writelines("{}: {}\n".format(name, value))
            reportfile.writelines(dedent("""
                            < Score >
                            F1: {}
                            Precision: {}
                            Recall: {}
                            """.format(f1_score(y, guess_y),
                                        precision_score(y, guess_y),
                                        recall_score(y, guess_y))
            ))
        
        _, ax = plt.subplots(figsize=(15,15))
        ConfusionMatrixDisplay(confusion_matrix(y, guess_y)).plot(ax=ax,cmap=plt.cm.Blues)
        ax.set_title(file_name)
        plt.savefig("./build/{}_confusion.png".format(file_name), bbox_inches="tight")
        plt.clf()

        fpr, tpr, _ = roc_curve(y, self.best_estimator_.predict_proba(self.vectorizer.transform(X))[:,1])
        plt.figure(figsize=(15, 15))
        plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (area = {:.2f})".format(auc(fpr, tpr)))
        plt.plot([0,1], [0,1], color="gray", linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} ROC Curve".format(file_name))
        plt.legend(loc="lower right")
        plt.savefig("./build/{}_roc.png".format(file_name), bbox_inches="tight")
        plt.clf()