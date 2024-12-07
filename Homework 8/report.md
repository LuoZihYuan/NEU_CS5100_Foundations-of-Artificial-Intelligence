# Homework 8
### Questions

1. Precision, recall, and F1 scores:
    - Naive Bayes by hand
        > F1: 0.96, Precision: 0.95, Recall: 0.97
    - Tfidf by hand
        > F1: 0.96, Precision: 1.00, Recall: 0.92
    - Naive Bayes using Scikit-learn
        > F1: 0.94, Precision: 0.95, Recall: 0.93
    - Tfidf using Scikit-learn
        > F1: 0.94, Precision: 0.97, Recall: 0.92 (norm=None)
    - Finetuned BERT
        > F1: 0.96, Precision: 1.00, Recall: 0.94 (2e-5)

2. Experiment with one of the parameters in the Tfidf vectorizer (character ngram analyzer, stop words, norm, etc.). Briefly explain what the parameter is, and show how it affected the scores when you changed it.
    >- ngram_range: Determine which n-gram to use. For example (1,3) means using all unigram, bigram and trigram. Using the more values for n doesn't guarantee a better score. However, skipping unigram would drasically worsen the recall and f1 scores.
    >- stop_words: Removes uninformative stop words such as "and", "the" to be used for prediction. Using this parameter would slightly decrease the recall score, thus effecting the F1 score.
    >- norm: "l1" means the sum of the vector components equals 1, "l2" means the sum of `the squares of` the vector components equals 1, and None means no normalization. Using None gives the best overall score.

3. How long did this assignment take you? (1 sentence)
    > 20 hours

4. Whom did you work with, and how? (1 sentence each)
    > No one. Just myself.

5. Which resources did you use? (1 sentence each)
    >- Class Recordings
    >- [Hugging Face Tutorial](https://huggingface.co/docs/transformers/training): Introduction to fine-tuning the Bert model

6. A few sentences about:
    - What was the most difficult part of the assignment?
        > Implement Naive Bayes by hand and ensuring its correctness.
    - What was the most rewarding part of the assignment?
        > Successfully fine-tuned Bert model and watch it achieve high overall score.
    - What did you learn doing the assignment?
        > The principle behind Naive Bayes and Tf-Idf.