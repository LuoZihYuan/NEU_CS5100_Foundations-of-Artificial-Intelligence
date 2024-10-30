# Homework 6
### Questions

1. In `solve_opponent_move()`, why do we return the opposite (negative) of the score used in the base case of `solve_my_move()`?
    > Because opponent would try to minimize the score.

2. Why is there no winner if both sides are playing optimally?
    > Cuz each of the positions on board have same weights, and when played optimally, both sides will always block each other.

3. This pseudocode includes alpha-beta pruning. What would be the pseudocode for implementing this same algorithm, but without alpha-beta pruning?
    > ```
    > if (game over):
    >   return -1 if player wins
    >   return 1 if opponent wins
    >   return 0 if tie
    > for empty spots in state:
    >   child := solve_opponent_move()
    >   best_move := (Max(best_move, child), empty spots)
    > return best_move
    > ```

4. How long did this assignment take you? (1 sentence)
    > 16 hours

5. Whom did you work with, and how? (1 sentence each)
    > No one. Just myself.

6. Which resources did you use? (1 sentence each)
    > Class Recordings

7. A few sentences about:
    - What was the most difficult part of the assignment?
        > Debugging what went wrong in the code.
    - What was the most rewarding part of the assignment?
        > Seeing each of the best_moves gets played.
    - What did you learn doing the assignment?
        > How to use alpha-beta pruning to greatly improve performance.