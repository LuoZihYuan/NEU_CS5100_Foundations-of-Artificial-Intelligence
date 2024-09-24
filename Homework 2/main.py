import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits

KFOLD = 5

def eval_clf(clf:BaseEstimator, grid:dict, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray) -> None:
    
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV
    
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    clf_cv = GridSearchCV(clf, grid, scoring=scorer, cv=KFOLD, n_jobs=-1, verbose=10)
    clf_cv.fit(train_x, train_y)
    best_clf = clf_cv.best_estimator_
    guess_y = best_clf.predict(test_x)

    print(f1_score(test_y, guess_y, average='weighted', zero_division=0))
    print(precision_score(test_y, guess_y, average='weighted', zero_division=0))
    print(recall_score(test_y, guess_y, average='weighted', zero_division=0))
    print(confusion_matrix(test_y, guess_y))

def classification_task():
    
    from _data import load_cats
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    x, y = load_cats()
    x = MinMaxScaler().fit_transform(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    receipes = [(
            LogisticRegression(penalty='l2', random_state=0, max_iter=200),
            {
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        ), (
            SVC(),
            {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        ), (
            DecisionTreeClassifier(),
            {
                'max_features': ['sqrt', 'log2'],
                'ccp_alpha': [0.1, .01, .001],
                'max_depth' : [5, 6, 7, 8, 9],
                'criterion' :['gini', 'entropy']
            }
        ), (
            MLPClassifier(activation='logistic', max_iter=400),
            {
                'hidden_layer_sizes': [
                    (48,), (96,), (144,), (192,),  # Single layer configurations
                    (48, 32), (96, 48), (144, 72),  # Two layers
                    (48, 32, 16), (96, 48, 24), (144, 72, 36),  # Three layers
                ],
                'solver': ['adam', 'sgd'],  # Test different solvers
                'alpha': [1e-4, 1e-3, 1e-2],  # L2 regularization strength
                'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
            }
        ), (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 4, 5, 10],
                'weights': ['uniform', 'distance'],
                'leaf_size': [15, 30, 60],
                'p': [1, 2, 3]
            }
        )
    ]
    
    for receipe in receipes:
        eval_clf(receipe[0], receipe[1], train_x, train_y, test_x, test_y)


def eval_regr(regr, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray) -> None:
    pass

def regression_task():
    
    from _data import load_age
    from sklearn.model_selection import train_test_split

def main():
    classification_task()
    regression_task()

if __name__ == "__main__":
    main()
