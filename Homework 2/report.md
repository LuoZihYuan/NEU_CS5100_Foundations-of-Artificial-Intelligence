# Homework 2
### Questions

1. For each classifier:
    - Report the precision, recall, and F1 score
    - Describe what the confusion matrix tells you about the performance (1 or 2 sentences)
    - Write a sentence or so about possible reasons why this model may or may not have been the right model for the task

    > | Confusion Matrix | Classifier | F1 | Precision | Recall | Performance Brief | Summary |
    > |:----------------:|:----------:|---:|----------:|-------:|:------------------|:--------|
    > |![](./build/bears/LogisticRegression.png)| Logistic Regression | 0.98 | 1.00 | 0.96 | Great Performance. Only 1 mistake. | It is the right model because the dataset is not a multinomial classification problem. |
    > |![](./build/bears/SVC.png)| Support Vector Machine | 1.00 | 1.00 | 1.00 | Great Performance. Only 1 mistake. | It is the right model because the dataset is not a multinomial classification problem. |
    > |![](./build/bears/DecisionTreeClassifier.png)| Decision Tree | 0.91 | 0.87 | 0.96 | Mediocre Performance. Some bears have been misidentified as pandas. | It is the right model because the dataset only contains 2 categoreis. |
    > |![](./build/bears/MLPClassifier.png)| Multi-layer Perceptron | 0.98 | 1.00 | 0.96 | Great Performance. Only 1 mistake. | It is the right model because the dataset is not a multinomial classification problem. |
    > |![](./build/bears/KNeighborsClassifier.png)| K-Nearest Neighbor | 0.98 | 1.00 | 0.96 | Great Performance. Only 1 mistake. | It is the right model because the dataset is not a multinomial classification problem. |

2. For each regression (numerical output) model:
    - Report the Mean Squared Error
    - Describe what the scatter plot tells you about the performance (1 sentence)
    - Write a sentence or so about possible reasons why this model may or may not have been the right model for the task

    > | Scatter Plot | Regressor | MSE | Performance Brief | Summary |
    > |:------------:|:---------:|----:|:------------------|:--------|
    > |![](./build/age/LinearRegression.png)| Linear Regression | 639.71 | Poor performance. Prediction is roughly 25 years older or younger than the label. | After flattening the images, the information of pixel distances have been lost. Thus, it might not be able to distinguish between an old man and a young girl. |
    > |![](./build/age/PolynomialRegression.png)| Polynomial Regression | 664.67 | Poor performance. Prediction is roughly 25 years older or younger than the label. | After flattening the images, the information of pixel distances have been lost. |
    > |![](./build/age/SVR.png)| Support Vector Machine | 4249199.87 | Poor performance. Prediction is roughly 25 years older or younger than the label. | After flattening the images, the information of pixel distances have been lost. |
    > |![](./build/age/MLPRegressor.png)| Multi-layer Perceptron | 381.57 | Poor performance. Prediction is slightly better, roughly 19 years older or younger than the label. | MLP performs slightly better than the other models. This is probably because the interconnection between nodes tries to restore the distance information of pixels. |
    > |![](./build/age/KNeighborsRegressor.png)| K-Nearest Neighbor | 455.89 | Poor performance. Prediction is slightly better, roughly 21 years older or younger than the label. | After flattening the images, the information of pixel distances have been lost. |

3. How long did this assignment take you? (1 sentence)
    > More than 32 hours.

4. Whom did you work with, and how? (1 sentence each)
    > Hsinyu Ko and I would have a little discussion about this assigment when we meet each other in class. We both encounter issues on predicting age from face images.

4. Which resources did you use? (1 sentence each)
    >- [GridSearchSV Tutorial](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/): Teaches how to use `GridSearchSV` to tune hyperparameters automatically.
    >- [Stack Overflow](https://stackoverflow.com/questions/57513586/how-to-use-grid-search-for-the-svm): Observe what hyerparameters are used by others to tune performance.

5. A few sentences about:
    - What was the most difficult part of the assignment?
        > Applying `PolynomialFeatures` on grayscale images pixels yields an explosion on total number of features, which has several times caused my computer to go OOM, then either stop responding or shutdown directly. I've had to try numerous methods to tackle this problem, such as scaling images down further more, or use `PCA`, `TruncatedSVD` to decrease the feature dimensions. But at last, I was only able to shrink down features by applying `SelectKBest`.
    - What was the most rewarding part of the assignment?
        > Changing my datasest from a 66 category classification to 2 category drastically improves the overall performance.
    - What did you learn doing the assignment?
        > That the technique of extracting features from images plays such an important role on determining the performance.