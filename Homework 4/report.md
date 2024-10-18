# Homework 4
### Questions
1. With a criteria function of lambda x: -len(x), did it encourage the model to generate shorter texts? Why or why not?
    > Yes, it does. The weights of normal characters decreases more than `end of sequence` characters.

2. Try out a criteria function of your choice. You could pass the predict()
function of a trained binary classifier, or a function which just prints the
text and asks you to specify (on a numerical scale) how much you like the
text. When testing it out, in this version of steps 4b and 4e, instead of
printing the average length, you’ll need to print the average score output
by this criteria function. Are you able to encourage the model to generate
texts in your preferred style?
    > Yes. My criteria function decreases the number of vowels appearing in the sentence. However, it also decreases the length of the generated sentence despite me encouraging longer sentences.

3. In Homework 1, the character ngram language model was using
frequencies as the weights when choosing the next letter, but for
Homework 5, we switched it to accommodate negative values as a result
of Q learning. One option we tried for this was to use the softmax of the
weights, which resulted in the generated songs becoming longer. Why?
    > Since softmax magnifies the differences between the overall distribution, and `end of sequence` character often appears less in a sentence. After applying softmax, the `end of sequence` character becomes more insignificant.

4. We also tried subtracting the minimum and adding 1 to each number.
Why did we add 1 instead of just subtracting the minimum?
    > If we only subtract the minimum without adding 1 back, the probability would become 0, which makes the letter impossible to appear.

5. In the generation stage, the difference in the length of generated songs
(as a result of Q learning) is much more dramatic if we test the same
prompt before and after Q learning. Why is the difference more dramatic
when we test the same prompt, as compared to using a different prompt
before and after Q learning?
    > If we use a different prompt, the model hasn't been optimized as much for that sequence, so the impact of Q-learning is less noticeable. The model behaves more like it did before Q-learning for unfamiliar prompts.

6. How long did this assignment take you? (1 sentence)
    > 13 hours

7. Whom did you work with, and how? (1 sentence each)
    > No one. Just myself.

8. Which resources did you use? (1 sentence each)
    > Class Recordings

9. A few sentences about:
    - What was the most difficult part of the assignment?
        > Finding a criteria function that works without significantly impacting the length of output.
    - What was the most rewarding part of the assignment?
        > Seeing the length of generated texts decreasing.
    - What did you learn doing the assignment?
        > The basics of unsupervised learning