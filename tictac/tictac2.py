from __future__ import print_function
 
import os
import string
from itertools import cycle
import random
import time
 
class TicTacGame(object):
    """class implements tictac"""
    def __init__(self, player1, player2):
        super(TicTac, self).__init__()
        self.board = [0] * 9
        self.players = [player1, player2]
        self.turn_order = cycle(self.players)
        self.current_player = next(self.players)

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
