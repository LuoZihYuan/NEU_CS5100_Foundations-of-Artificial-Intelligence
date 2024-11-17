# Homework 7
### Questions

1. How well do the topics represent real-world topics?
    > After examining the topics with respect to their top words, I'd say it does a pretty good job at representing real-world topics. First, the top words of the same topic are closely related to each other, which gives a clear picture of what the topics are about. Second, although most of the topics are related to politics, we can still easily tell the difference between each topic.

2. Which topics are prevalent in the real news documents? Which topics are prevalent in the fake news documents?
    > According to the image below, we can find that topic 9 is more prevalent in real news documents, while topics 1 and 7 are more prevalent in fake news documents.
    > ![](./build/03_topic_dist.png)

3. According to the resulting coefficients from the regression, which topics are most useful in determining whether something is real news or fake news?
    > Topic 9 is the most useful among all topics, while topics 5, 0 and 2 comes next in terms of importance.

4. Select 5 news documents from each resulting cluster. Do the clusters correspond to anything?
    > Yes, especially if the 5 documents' lda vectors are the closest to the centroid.

5. How long did this assignment take you? (1 sentence)
    > 10 hours

6. Whom did you work with, and how? (1 sentence each)
    > No one. Just myself.

7. Which resources did you use? (1 sentence each)
    > Class Recordings

8. A few sentences about:
    - What was the most difficult part of the assignment?
        > To get the top 5 vectors nearest to the centroid of each KMeans cluster.
    - What was the most rewarding part of the assignment?
        > Seeing that logistic regression gives a f1 score of 90+%.
    - What did you learn doing the assignment?
        > The basic usages of pandas, scipy, numpy and matplotlib