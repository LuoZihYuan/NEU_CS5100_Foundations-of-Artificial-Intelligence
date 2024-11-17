import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import LatentDirichletAllocation

def eval_clf(clf:BaseEstimator, grid:dict, train_x: np.ndarray, train_y:np.ndarray, test_x:np.ndarray, test_y:np.ndarray) -> None:
    
    from textwrap import dedent
    from matplotlib import pyplot as plt
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.model_selection import GridSearchCV
    
    scorer = make_scorer(f1_score)

    clf_cv = GridSearchCV(clf, grid, scoring=scorer, cv=5, n_jobs=-1, verbose=10)
    clf_cv.fit(train_x, train_y)
    best_clf = clf_cv.best_estimator_

    coefficients = best_clf.coef_[0]
    odds_ratios = np.exp(coefficients)
    importance_df = pd.DataFrame({
        "Topic": range(10),
        "Coefficient": coefficients,
        "Odds Ratio": odds_ratios
    })

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
        reportfile.writelines("< Topic Importance >\n")
        reportfile.writelines(importance_df.sort_values(by="Coefficient", ascending=False).to_string(index=False))

    cm = confusion_matrix(test_y, guess_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    _, ax = plt.subplots(figsize=(15,15))
    disp.plot(ax=ax,cmap=plt.cm.Blues)
    disp.ax_.set_title(clf_name)
    plt.savefig("./build/{}.png".format(clf_name), bbox_inches='tight')
    plt.clf()

def Q2(lda:LatentDirichletAllocation, lexicon:np.ndarray):
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([lexicon[i] for i in topic.argsort()[:-20 - 1:-1]]))

def Q3(all_lda:np.ndarray, fake_size:int, real_size:int):
    
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    BAR_W = 0.8
    SAMPLE_SIZE = 5
    TOPIC_SIZE = all_lda.shape[1]

    fig, (splt1, splt2) = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    fig.set_dpi(300)

    sample_fake = np.random.choice(fake_size,SAMPLE_SIZE,replace=False)
    sample_fake_lda = all_lda[:fake_size,:][sample_fake,:]
    for i in range(TOPIC_SIZE):
        splt1.bar([j + i/TOPIC_SIZE * BAR_W for j in range(SAMPLE_SIZE)], sample_fake_lda[:,i], width=BAR_W/TOPIC_SIZE, label=str(i))
    splt1.title.set_text("fake")

    sample_real = np.random.choice(real_size,SAMPLE_SIZE,replace=False)
    sample_real_lda = all_lda[-real_size:,:][sample_real,:]
    for i in range(TOPIC_SIZE):
        splt2.bar([j + i/TOPIC_SIZE * BAR_W for j in range(SAMPLE_SIZE)], sample_real_lda[:,i], width=BAR_W/TOPIC_SIZE, label=str(i))
    splt2.title.set_text("real")
    
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig.legend(handles=[Patch(color=plt_colors[i]) for i in range(TOPIC_SIZE)], labels=[i for i in range(TOPIC_SIZE)], title="topic", loc='upper center', ncol=TOPIC_SIZE)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("./build/03_topic_dist.png", dpi=300)
    plt.clf()

def Q4(all_lda:np.ndarray, fake_size:int, real_size:int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    x = all_lda
    y = pd.concat([pd.Series(0,index=range(fake_size)), pd.Series(1, index=range(real_size))], ignore_index=True)
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

    clf = LogisticRegression(penalty='l2', random_state=0, max_iter=200)
    grid = {
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
    eval_clf(clf, grid, train_x, train_y, test_x, test_y)

def Q5(all_lda:np.ndarray, all_news:pd.Series, fake_size:int, real_size:int):
    
    from scipy.spatial import distance
    from sklearn.cluster import KMeans

    fake_lda = all_lda[:fake_size,:]
    fake_news = all_news[:fake_size]
    cls = KMeans(n_clusters=10).fit(fake_lda)

    for i in range(10):
        print(i)
        centroid = cls.cluster_centers_[i]
        element_index = np.where(cls.labels_ == i)[0]
        nearest_index = element_index[distance.cdist([centroid], fake_lda[element_index])[0].argsort()[:5]]
        for j in nearest_index:
            print(fake_news[j])
        print()

def main():
    import joblib
    import os.path
    from sklearn.feature_extraction.text import CountVectorizer

    fake_news_pd = pd.read_csv("./data/Fake.csv", usecols=['text'])
    fake_news = fake_news_pd["text"]
    fake_size = fake_news.shape[0]
    real_news_pd = pd.read_csv("./data/True.csv", usecols=['text'])
    real_news = real_news_pd["text"]
    real_size = real_news.shape[0]
    all_news = pd.concat([fake_news,real_news], ignore_index=True)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, lowercase=True, stop_words='english')
    all_tf = tf_vectorizer.fit_transform(all_news)
    lexicon = tf_vectorizer.get_feature_names_out()
    if os.path.isfile("./data/lda.pkl"):
        lda = joblib.load("./data/lda.pkl")
    else:
        lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(all_tf)
        joblib.dump(lda, "./data/lda.pkl")

    all_lda = lda.transform(all_tf)
    
    Q2(lda, lexicon)
    Q3(all_lda, fake_size, real_size)
    Q4(all_lda,fake_size,real_size)
    Q5(all_lda,all_news,fake_size,real_size)

if __name__ == "__main__":
    main()