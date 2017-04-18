from __future__ import print_function

import os
import string
from itertools import cycle
import random
from neat import nn, population, statistics
import time
import pickle

 
class tictac(object):
    """class implements tictac"""
    def __init__(self, net=None, rand=False):
        super(tictac, self).__init__()
        self.boards = [[0] * 9, [0] * 9]
        self.moves_remaining = 9
        self.players = cycle(['X','O'])
        self.winner = False
        self.current_player=next(self.players)
        self.net = net
        self.rand=rand

 
    def check_vector(self, iterator):
        # given a segment, are all the positions the same.
        return bool(bool(len(set(iterator)) == 1) and iterator[0])
 
    def visual(self):
        rows=[0,3,6]
        self.moves_remaining = 8
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

            print(ascii_row)


    def itemgetter(self, indicies, iterator):
        return map(iterator.__getitem__, indicies)
 
    def is_valid_play(self, move):
        for board in self.boards:
            if board[move]:
                return False
        return True
 
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
                self.itemgetter([2, 4, 6], board)]
            for vector in vectors:
                if self.check_vector(vector):
                    return self.current_player
        if self.moves_remaining <= 0:
            return "No One"
        return False

    def set_move(self, play):
        if self.current_player == 'X':
            self.boards[1][play] = 1
        else:
            self.boards[0][play] = 1


    def doturn(self, play=None):
        self.current_player = next(self.players)
        while play==None:
            if self.net:
                if self.current_player=='X':
                    inputs = []
                    inputs = list(self.boards[0])
                    inputs.extend(self.boards[1])
                    votes={}
                    output = self.net.serial_activate(inputs)
                    for idx, val in enumerate(output):
                        votes[idx] = val
                    votes = sorted(votes, key=votes.get)
                    for vote in votes:
                        if self.is_valid_play(vote):
                            play = vote
                            self.set_move(play)
                            print ("Net plays: {}".format(play))
                            return
            if self.rand:
                if self.current_player=='O':
                    votes=[0,1,2,3,4,5,6,7,8]
                    random.shuffle(votes)
                    for vote in votes:
                        if self.is_valid_play(vote):
                            play = vote
                            self.set_move(play)
                            print ("Rnd plays: {}".format(play))
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
        print("Game Over!, {} Wins!".format(self.winner))
        return self.winner

with open('winner.pkl', 'rb') as input:
    winner = pickle.load(input)

winner_net = nn.create_feed_forward_phenotype(winner)

game = tictac(winner_net,rand=False)
result = game.start_game()
