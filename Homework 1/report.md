# Homework 1
### Questions

1. Describe how the choice of training dataset affected your model. What happens if we train a model on one type of data (like music lyrics) and then ask it to work with a different type of data (such as medical reports)? Which datasets do popular language models use, and how might this affect their output? (1 paragraph)
    >- The selection of datasets drastically effects the outcome of the language model. For example, if the langauge model is trained on music lyrics, supposedly, it kwows nothing about medical reports. Thus, when it's used to predict words for medical reports, rather than following the contexts, it would just keep spitting out music lyrics.
    >- Most of the well-know language models like GPT-3, LLaMA use datasets like "CommonCrawl", "The Pile", "Wikipedia", etc. The problem with these datasets, specifically "CommonCrawl", is that most of the data are not from trustful sources, thus they are prone to give untruthful answers.

2. How long did this assignment take you? (1 sentence)
    > Roughly 1.5 hours to implement the original character level n-gram prediction model, another 0.5 hours to implement word level n-gram prediction model and bi-directional data processing.

3. Whom did you work with, and how? (1 sentence each)
    > Hsinyu Ko and I would have a little discussion about this assigment when we meet each other in class. Originally, I thought the `generate_word` method should always return the character with the highest probability score, yet she corrected me by saying the results would end up the same everytime.

4. Which resources did you use? (1 sentence each)
    >- [Stack Overflow](https://stackoverflow.com/a/41852266/27310118): Ways to randomly pick a character based on its probability distribution.
    >- [Stack Overflow](https://stackoverflow.com/a/55481809/27310118): Ways to turn dictionary values and keys into list

5. A few sentences about:
    - What was the most difficult part of the assignment?
        > The amount of time it takes to search for suitable datasets and cleanup data.
    - What was the most rewarding part of the assignment?
        > That one time when the model is able to generate a complete and resonable sentence.
    - What did you learn doing the assignment?
        > That datasets really effect the quality of the outcome. Small datasets would often encounter KeyError, especially when n turns bigger and bigger.