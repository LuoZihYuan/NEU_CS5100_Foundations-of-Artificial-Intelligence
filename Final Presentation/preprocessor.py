from __future__ import annotations

import pandas as pd
from re import sub, MULTILINE
from datetime import datetime
from nltk import download, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

class FAIBasePreprocessor:

    def __init__(self,
                 lowercase:bool=False,
                 rm_markup:bool=False,
                 rm_url:bool=False,
                 rm_non_word:bool=False,
                 lemmatize:bool=False,
                 rm_stopword:bool=False) -> None:
        
        self.lowercase = lowercase
        self.rm_markup = rm_markup
        self.rm_url = rm_url
        self.lemmatize = lemmatize
        self.rm_stopword = rm_stopword
        self.rm_non_word = rm_non_word

    def transform(self, X:pd.Series, y:pd.Series) -> tuple:
        print("Start <Preprocess>")
        processed_X = X
        if self.lowercase:
            processed_X = self._lower(processed_X)
        if self.rm_markup:
            processed_X = self._rm_markup(processed_X)
        if self.rm_url:
            processed_X = self._rm_url(processed_X)
        if self.rm_non_word:
            processed_X = self._rm_non_word(processed_X)
        if self.lemmatize:
            processed_X = self._lemmatize(processed_X)
        if self.rm_stopword:
            processed_X = self._rm_stop_word(processed_X)

        processed_Y = y.apply(lambda label: label if not label else 1)

        return processed_X, processed_Y

    def _lower(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Lower (lowercase all tweets)".format(datetime.now()))
        return X.apply(lambda txt: txt.lower())
    
    def _rm_markup(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Remove Markup (remove markup tags from all tweets)".format(datetime.now()))
        rgx_markup = r"<.*?>"
        return X.apply(lambda txt: sub(rgx_markup, "", txt, flags=MULTILINE))
    
    def _rm_url(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Remove URL".format(datetime.now()))
        rgx_url = r"(?:https?://)?(?:www\.)?(?:[0-9][a-zA-Z0-9]+|[a-zA-Z][a-zA-Z0-9]*)(?:\.[a-zA-Z0-9]{2,}){1,}[^\s]*"
        return X.apply(lambda txt: sub(rgx_url, "", txt, flags=MULTILINE))
    
    def _rm_non_word(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Remove Non-word".format(datetime.now()))
        rgx_non_word = r"[\W]"
        return X.apply(lambda txt: sub(rgx_non_word, " ", txt, flags=MULTILINE))

    def _lemmatize(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Lemmatize (turn words into their original form)".format(datetime.now()))
        download('wordnet')
        download('punkt_tab')
        download('averaged_perceptron_tagger_eng')

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        def parse_pos(pos:str):
            if pos.startswith("J"):
                return wordnet.ADJ
            elif pos.startswith("V"):
                return wordnet.VERB
            elif pos.startswith("R"):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        return X.apply(lambda txt: " ".join([stemmer.stem(lemmatizer.lemmatize(word, parse_pos(pos))) for word, pos in pos_tag(word_tokenize(txt))]))

    def _rm_stop_word(self, X:pd.Series) -> pd.Series:
        print("{} <Preprocess> Remove Stopword".format(datetime.now()))
        download("stopwords")

        return X.apply(lambda txt: " ".join([word for word in word_tokenize(txt) if word not in stopwords.words("english")]))
    

class FAINaiveBayesPreprocessor(FAIBasePreprocessor):
    def __init__(self):
        super().__init__(lowercase=True,
                         rm_markup=True,
                         rm_url=True,
                         lemmatize=True,
                         rm_stopword=True,
                         rm_non_word=True)

class FAILogisticRegressionPreprocessor(FAIBasePreprocessor):
    def __init__(self):
        super().__init__(lowercase=True,
                         rm_markup=True,
                         rm_url=True,
                         lemmatize=True,
                         rm_stopword=True,
                         rm_non_word=True)

class FAIAdaBoostPreprocessor(FAIBasePreprocessor):
    def __init__(self):
        super().__init__(lowercase=True,
                         rm_markup=True,
                         rm_url=True,
                         lemmatize=True,
                         rm_stopword=True,
                         rm_non_word=True)

class FAIXGBoostPreprocessor(FAIBasePreprocessor):
    def __init__(self):
        super().__init__(lowercase=True,
                         rm_markup=True,
                         rm_url=True,
                         lemmatize=True,
                         rm_stopword=True,
                         rm_non_word=True)

class FAIBertPreprocessor(FAIBasePreprocessor):
    def __init__(self) -> None:
        super().__init__(lowercase=True)