# Homework 5
### Questions

1. Please give an example of a word which was correctly spelled by the user, but which was incorrectly “corrected” by the algorithm. Why did this happen?
    > `assignment` is incorrectly corrected to `assigement`. This might be affected by the transition matrix, since it tends to favor frequent character transitions.

2. Please give an example of a word which was incorrectly spelled by the user, but which was still incorrectly “corrected” by the algorithm. Why did this happen?
    > `teh` is incorrectly corrected to `th` instead of `the`. This might still be affected by the transition matrix, since it tends to favor frequent character transitions.

3. Please give an example of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm. Why was this one correctly corrected, while the previous two were not?
    > `He` is correctly corrected to `Hi`. This might be due to the fact that `i` is commonly transitioned from `H`.

4. How long did this assignment take you? (1 sentence)
    > 4 hours

5. Whom did you work with, and how? (1 sentence each)
    > No one. Just myself.

6. Which resources did you use? (1 sentence each)
    > Class Recordings

7. A few sentences about:
    - What was the most difficult part of the assignment?
        > Calculating the lattice metric
    - What was the most rewarding part of the assignment?
        > Seeing the output of the `correct` method.
    - What did you learn doing the assignment?
        > The Viterbi algorithm