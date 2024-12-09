import pandas as pd
from pipeline import FAIPipeline
from preprocessor import FAINaiveBayesPreprocessor

def main():
    df = pd.read_csv("./data/training.1600000.processed.noemoticon.csv", names=["target", "ids", "date", "flag", "user", "text"], usecols=["text", "target"], encoding='ISO-8859-1')\
        #  .sample(80000, ignore_index=True)

if __name__ == "__main__":
    main()