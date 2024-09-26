import numpy as np
from sklearn.base import BaseEstimator

KFOLD = 5

def eval_clf(clf:BaseEstimator, grid:dict, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray) -> None:
    
    from textwrap import dedent
    from matplotlib import pyplot as plt
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import GridSearchCV
    
    scorer = make_scorer(f1_score)

    clf_cv = GridSearchCV(clf, grid, scoring=scorer, cv=KFOLD, n_jobs=-1, verbose=10)
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
                           Recall: {}
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

def classification_task():
    
    from _data import load_bears
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    x, y = load_bears()
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
                    (48, 32, 16), (96, 48, 24), (144, 72, 36)  # Three layers
                ],
                'solver': ['adam', 'sgd'],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate': ['constant', 'adaptive']
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


def eval_regr(regr:BaseEstimator, grid:dict, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray, name:str = "", n_jobs:int=-1) -> None:
    
    from textwrap import dedent
    from matplotlib import pyplot as plt
    from sklearn.metrics import make_scorer, mean_squared_error
    from sklearn.model_selection import GridSearchCV

    scorer = make_scorer(mean_squared_error)
    regr_cv = GridSearchCV(regr, grid, scoring=scorer, cv=KFOLD, n_jobs=n_jobs, verbose=10)
    regr_cv.fit(train_x, train_y)
    best_regr = regr_cv.best_estimator_
    guess_y = best_regr.predict(test_x)

    regr_name = type(regr).__name__ if not name else name
    with open("./build/{}.txt".format(regr_name), "w") as reportfile:
        reportfile.writelines("{}\n".format(regr_name))
        reportfile.writelines(dedent("""
                              < Best Parameters >
                              """
        ))
        for name, value in regr_cv.best_params_.items():
            reportfile.writelines("{}: {}\n".format(name, value))
        reportfile.writelines(dedent("""
                           < Score >
                           MSE: {}
                           """.format(mean_squared_error(test_y, guess_y))
        ))
    plt.scatter(guess_y, test_y)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(regr_name)
    plt.savefig("./build/{}.png".format(regr_name), bbox_inches='tight')
    plt.clf()

def regression_task():
    
    from _data import load_age
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_regression
    
    x, y  = load_age()
    x = MinMaxScaler().fit_transform(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    receipes = [(
            LinearRegression(),
            {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        ), (
            Lasso(),
            {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        ), (
            Ridge(),
            {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        ), (
            SVR(),
            {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        ), (
            MLPRegressor(activation='relu', max_iter=400),
            {
                'hidden_layer_sizes': [
                    (48,), (96,), (144,), (192,),  # Single layer configurations
                    (48, 32), (96, 48), (144, 72),  # Two layers
                    (48, 32, 16), (96, 48, 24), (144, 72, 36)  # Three layers
                ],
                'solver': ['adam', 'sgd'],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate': ['constant', 'adaptive']
            }
        ), (
            KNeighborsRegressor(),
            {
                'n_neighbors': [3, 4, 5, 10],
                'weights': ['uniform', 'distance'],
                'leaf_size': [15, 30, 60],
                'p': [1, 2, 3]
            }
        )
    ]
    for receipe in receipes:
        eval_regr(receipe[0], receipe[1], train_x, train_y, test_x, test_y)

    x = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(x)
    x = SelectKBest(score_func=f_regression, k=50000).fit_transform(x, y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    eval_regr(receipes[0][0], receipes[0][1], train_x, train_y, test_x, test_y, name="PolynomialRegression", n_jobs=1)

def main():
    classification_task()
    # regression_task()

if __name__ == "__main__":
    main()
