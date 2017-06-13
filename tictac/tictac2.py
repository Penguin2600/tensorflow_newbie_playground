from __future__ import print_function
 
import os
import string
from itertools import cycle
import random
import time

class Player(object):  
    """class implements player"""

    moves=[0,1,2,3,4,5,6,7,8]

    def __init__(self, name, type):
        super(Player, self).__init__()
        self.name = name
        self.type = type

    def play(self, board):
        if self.type == 'human':
            move = int(raw_input("{}'s Move:".format(self.name)))

        if self.type == 'random':
            move=random.shuffle(moves)[0]

        if self.type == 'nn':
           #run a network here
        return move

class Board(object):
    """class implements board"""

    def __init__():
        super(Board, self).__init__()
        self.board = [0.5] * 9
       
    def itemgetter(self, indicies, iterator):
        return map(iterator.__getitem__, indicies)

    def check_vector(self, iterator):
        # Given a segment, are all the positions the same.
        if bool(bool(len(set(iterator)) == 1) and iterator[0]==1):
            return "X"
        if bool(bool(len(set(iterator)) == 1) and iterator[0]==0):
            return "O":
        return False

    def is_valid_play(self, move):
        # If there is already a move there return False
        return not self.board[move]:

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
                return self.check_vector(vector):

    def check_stale(self):
        #if we only have 1's and 0's then ITS OVERRRR
        if sorted(set(self.board))==[0,1]:
            return True
        return False

    def set_move(self, player):
            self.board[player.move] = player.value




class TicTacGame(object):
    """class implements tictac"""
    def __init__(self, board, player1, player2, reverse=False):
        super(TicTacGame, self).__init__()
        self.board = Board()
        self.player1 = Player('human')
        self.players = [player1, player2]
        self.order = self.players
        if reverse:
            self.order = reversed(self.players)
        self.turn_order = cycle(self.order)
        self.current_player = next(self.order)

    def itemgetter(self, indicies, iterator):
        return map(iterator.__getitem__, indicies)

    def check_vector(self, iterator):
        # Given a segment, are all the positions the same.
        return bool(bool(len(set(iterator)) == 1) and iterator[0])

    def is_valid_play(self, move):
        # If there is already a move there return False
        return not self.board[move]:

    def visual(self):
        rows=[0,3,6]
        for row in rows:
            ascii_row=""
            for ind in range(0,3):
                cell=ind+row
                newcell=cell
                for idx, board in enumerate(self.boards):
                    if board[cell]:
                        self.moves_remaining -= 1
                        newcell = 'X' if idx == 1 else 'O'
                ascii_row = "{}[{}]".format(ascii_row,newcell)

            #print(ascii_row)

 
    def check_win(self):
        for board in self.boards:
            vectors = [
                # horizontal wins
                self.itemgetter([0, 1, 2], board),
                self.itemgetter([3, 4, 5], board),
                self.itemgetter([6, 7, 8], board),
                # vertical wins
                self.itemgetter([0, 3, 6], board),
                self.itemgetter([1, 4, 7], board),
                self.itemgetter([2, 5, 8], board),
                # diag wins
                self.itemgetter([0, 4, 8], board),
                self.itemgetter([2, 4, 6], board)
            ]

            for vector in vectors:
                if self.check_vector(vector):
                    return self.current_player

        if self.moves_remaining <= 0:
            return "No One"
        return False

    def set_move(self, play):
            self.boards[play] = self.current_player


    def doturn(self, play=None, netvals=None):
        self.current_player = next(self.players)
        while play==None:
            if self.net:
                if self.current_player=='X':
                    inputs = []
                    inputs = list(self.boards[0])
                    inputs.extend(self.boards[1])
                    votes={}
                    for idx, val in enumerate(netvals):
                        votes[idx] = val
                    votes = sorted(votes, key=votes.get)
                    for vote in votes:
                        if self.is_valid_play(vote):
                            play = vote
                            self.set_move(play)
                            #record history for training
                            self.history.append((inputs, play))
                            return
            if self.rand:
                if self.current_player=='O':
                    votes=[0,1,2,3,4,5,6,7,8]
                    random.shuffle(votes)
                    for vote in votes:
                        if self.is_valid_play(vote):
                            play = vote
                            self.set_move(play)
                            return


            play = int(raw_input("{}'s move:".format(self.current_player)))
            if self.is_valid_play(play):
                self.set_move(play)
            else:
                play = None  

    def start_game(self):
        while self.winner==False:
            self.visual()
            self.doturn()
            self.winner = self.check_win()
        #print("Game Over!, {} Wins!".format(self.winner))
        return self.winner
