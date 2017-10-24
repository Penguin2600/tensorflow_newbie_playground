from __future__ import print_function
 
import os
import string
from itertools import cycle
import random
import time

class Player(object):  
    """class implements player"""

    def __init__(self, name, type, value):
        super(Player, self).__init__()
        self.name = name
        self.type = type
        self.move = None
        self.value = value
        self.rand_state=[0,1,2,3,4,5,6,7,8]

    def play(self, board):
        if self.type == 'human':
            is_valid=None
            while not is_valid:
                self.move = int(input("{}'s Move:".format(self.name)))
                is_valid = board.set_move(self.move, self.value)

        if self.type == 'random':
            random.shuffle(self.rand_state)
            is_valid=None
            move_index=0
            while not is_valid:
                self.move = self.rand_state[move_index]
                is_valid = board.set_move(self.move, self.value)
                move_index+=1

        if self.type == 'nn':
            #run a network here
            return move

class Board(object):
    """class implements board"""

    def __init__(self):
        super(Board, self).__init__()
        self.board = [0.5] * 9
      

    def is_valid_play(self, move):
        # If there is already a move there return False
        return (self.board[move] == 0.5)

    def set_move(self, location, value):
        is_valid = self.is_valid_play(location)
        if is_valid:
            self.board[location] = value
        return is_valid

    def do_visual(self):
        rows=[0,3,6]
        for row in rows:
            ascii_row=""
            for ind in range(0,3):
                cell=ind+row
                newcell = ' '
                if self.board[cell]==0:
                    newcell = 'O'
                if self.board[cell]==1:
                    newcell = 'X'
                ascii_row = "{}[{}]".format(ascii_row,newcell)
            print(ascii_row)

    def check_vector(self, iterator):
        # Given a segment, are all the positions the same.
        if len(set(iterator)) == 1 and iterator[0] != 0.5:
            return True
        return False

    def itemgetter(self, indicies, iterator):
        result = [iterator[i] for i in indicies]
        return result

    def check_win(self):
            vectors = [
                # horizontal wins
                self.itemgetter([0, 1, 2], self.board),
                self.itemgetter([3, 4, 5], self.board),
                self.itemgetter([6, 7, 8], self.board),
                # vertical wins
                self.itemgetter([0, 3, 6], self.board),
                self.itemgetter([1, 4, 7], self.board),
                self.itemgetter([2, 5, 8], self.board),
                # diag wins
                self.itemgetter([0, 4, 8], self.board),
                self.itemgetter([2, 4, 6], self.board)
            ]

            for vector in vectors:
                if self.check_vector(vector):
                    return True

            if self.check_stale():
                return False

            return None

    def check_stale(self):
        #if we only have 1's and 0's then ITS OVERRRR
        if sorted(set(self.board))==[0,1]:
            return True
        return False

class TicTacGame(object):
    """class implements tictac"""
    def __init__(self, board, player1, player2):
        super(TicTacGame, self).__init__()
        self.board = Board()
        self.players = [player1, player2]
        self.turn_order = cycle(self.players)
        self.current_player = None
        self.winner = None

    def run_game(self):
        while self.winner==None:
            self.current_player = next(self.turn_order)
            self.board.do_visual()
            self.current_player.play(self.board)
            self.winner = self.board.check_win()
        if self.winner:
            print("Game Over!, {} Wins!".format(self.current_player.name))
        else:
            print("Game Over!, No one Wins!")
        return self.winner, self.current_player

if __name__ == "__main__":
    #Do a standard human human game.
    p1=Player('p1', 'human', 1)
    p2=Player('p2', 'random', 0)
    board=Board()
    localgame=TicTacGame(board, p1, p2)
    localgame.run_game()
