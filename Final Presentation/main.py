import pandas as pd
from model import FAINaiveBase, FAILogisticRegression, FAIAdaBoost, FAIXGBoost
from preprocessor import FAIBasePreprocessor
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("./data/training.1600000.processed.noemoticon.csv", names=["target", "ids", "date", "flag", "user", "text"], usecols=["text", "target"], encoding='ISO-8859-1')\
        #  .sample(80000, ignore_index=True)
    pp = FAIBasePreprocessor(lowercase=True,
                             rm_markup=True,
                             rm_url=True,
                             lemmatize=True,
                             rm_stopword=True,
                             rm_non_word=True)
    df["text"], df["target"] = pp.transform(df["text"], df["target"])
    # df["target"] = df["target"].apply(lambda label: label if not label else 1)
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=True)
    FAINaiveBase().fit(df_train["text"], df_train["target"])\
                  .evaluate(df_test["text"], df_test["target"])
    FAILogisticRegression().fit(df_train["text"], df_train["target"])\
                           .evaluate(df_test["text"], df_test["target"])
    FAIAdaBoost().fit(df_train["text"], df_train["target"])\
                 .evaluate(df_test["text"], df_test["target"])
    FAIXGBoost().fit(df_train["text"], df_train["target"])\
                .evaluate(df_test["text"], df_test["target"])

if __name__ == "__main__":
    main()