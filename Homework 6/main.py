from __future__ import annotations # for type hints in recursive node structure
from enum import Enum

# class Counter:
#     def __init__(self) -> None:
#         self.count = 0
    
#     def add(self) -> None:
#         self.count += 1
#         print(self.count)
# universal_counter = Counter()

class Move:
    INVALID_COORDINATE = -1
    def __init__(self, value:float, row:int=INVALID_COORDINATE, column:int=INVALID_COORDINATE) -> None:
        self.row = row
        self.column = column
        self.value = value

class Player(Enum):
    O = "O"
    X = "X"
    def __str__(self) -> str:
        return self.value
    
    def opponent(self) -> Player:
        return Player.X if self.value == "O" else Player.O

class GameState:
    def __init__(self) -> None:
        self.board = [[None, None, None],
                      [None, None, None],
                      [None, None, None]]
    
    def __str__(self) -> str:
        board_str = [[" " if not j else str(j) for j in i] for i in self.board]
        return "\n---------\n".join([" | ".join(row) for row in board_str])

    def winner(self) -> Player:
        if self.game_over():
            return self._winner
        return None
    
    def game_over(self) -> bool:
        self._winner = None
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != None:
            self._winner = self.board[0][0]
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != None:
            self._winner = self.board[0][2]
            return True
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != None:
                self._winner = self.board[i][0]
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != None:
                self._winner = self.board[0][i]
                return True
        empty_spots = sum([1 if not j else 0 for i in self.board for j in i])
        return False if empty_spots > 0 else True
    
    def spot(self, row:int, col:int) -> Player:
        return self.board[row][col]
    
    def move(self, row:int, col:int, player:Player) -> GameState:
        if self.board[row][col]:
            return None
        else:
            from copy import deepcopy
            next_state = GameState()
            next_state.board = deepcopy(self.board)
            next_state.board[row][col] = player
            return next_state

class TicTacToeSolver:
    def find_best_move(self, state:GameState, player:Player) -> Move:
        return self.solve_my_move(state, player)

    def solve_my_move(self, state:GameState, player:Player, alpha:float=float('-inf'), beta:float=float('inf')) -> Move:
        if state.game_over():
            if state.winner() == None:
                return Move(0)
            elif state.winner() == player:
                return Move(1)
            else:
                return Move(-1)
        best_move:Move = None
        for i in range(3):
            for j in range(3):
                next_state = state.move(i,j,player)
                if next_state:
                    child = self.solve_opponent_move(next_state, player.opponent(), alpha=alpha, beta=beta)
                    if not best_move or child.value > best_move.value:
                        best_move = Move(child.value, i, j)
                        alpha = best_move.value
                        if alpha > beta:
                            return best_move
        return best_move
    
    def solve_opponent_move(self, state:GameState, player:Player, alpha:float=float('-inf'), beta:float=float('inf')) -> Move:
        if state.game_over():
            if state.winner() == None:
                return Move(0)
            elif state.winner() == player:
                return Move(-1)
            else:
                return Move(1)
        best_move:Move = None
        for i in range(3):
            for j in range(3):
                next_state = state.move(i,j,player)
                if next_state:
                    child = self.solve_my_move(next_state, player.opponent(), alpha=alpha, beta=beta)
                    if not best_move or child.value < best_move.value:
                        best_move = Move(child.value, i, j)
                        beta = best_move.value
                        if alpha > beta:
                            return best_move
        return best_move


def main():
    current_state = GameState()
    print(current_state)
    print()
    current_player = Player.X
    solver = TicTacToeSolver()
    while not current_state.game_over():
        best_move = solver.find_best_move(current_state, current_player)
        i,j = best_move.row, best_move.column
        current_state = current_state.move(i, j, current_player)
        current_player = current_player.opponent()
        print(current_state)
        print()

if __name__ == "__main__":
    main()