from __future__ import print_function


#Supress TF Spam
import os
from copy import copy
from tictac2 import TicTacGame, Player, Board
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Some globals :/ dont hate.

do_training = 1 # 1 = do the training, 0 = load from file and just run it
save_trained = 1 # 1 = save to file after training, 0 = don't save
# Change the following to where you want the network to be saved to.
# Make sure to create the directory structure.
save_file = './tflogdir/model.ckpt'
write_for_tensorboard = 1 # 1 = write info for TensorBoard, 0 = don't
# Change the following to where you want the info to be saved.
# Make sure to create the directory structure.
tensorboard_file = './tflogdir/tftictac/1'

NUM_INPUTS = 9
NUM_HIDDEN = 9
NUM_OUTPUTS = 9
ITTERCOUNT = 500

NUM_IN_TRAINING_SET = 0

inputvals  = []
targetvals = []


import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS], name='y_')

def vote_for(move):
    vote = [0]*9
    vote[move] = 1
    return vote

# play a bunch of games and remember the ones we won
def play_games_random():
    won_games=[]
    with tf.name_scope('play_game'):
        for _ in range(500):
            #generate new game
                #Do a standard human random game.
            p1=Player('random_player1', 'random', 1)
            p2=Player('random_player2', 'random', 0)
            board=Board()
            localgame=TicTacGame(board, p1, p2)
            history=[]

            while localgame.winner==None:
                localgame.current_player = next(localgame.turn_order)
                localgame.board.do_visual()
                is_valid = localgame.current_player.play(localgame.board)
                if localgame.current_player.name=='random_player1':
                  history.append((copy(localgame.board.board), localgame.current_player.move))
                localgame.winner = localgame.board.check_win()
            if localgame.winner:
                print("Game Over!, {} Wins!".format(localgame.current_player.name))
                if localgame.current_player.name == 'random_player1':
                  #take winning games and build training data
                  won_games.append(history)
                  print(history)
            else:
                print("Game Over!, No one Wins!")

        print(len(won_games))
        for game in won_games:
            for move in game:
                inputvals.append(move[0])
                targetvals.append(vote_for(move[1]))
        print(len(won_games)/float(ITTERCOUNT))
# play a bunch of games and remember the ones we won
def play_human():
    won_games=[]
    with tf.name_scope('play_game'):
        for _ in range(ITTERCOUNT):
            #generate new game
                #Do a standard human random game.
            p1=Player('ai_player', 'AI', 1)
            p2=Player('random_player', 'human', 0)
            board=Board()
            localgame=TicTacGame(board, p1, p2)
            history=[]

            while localgame.winner==None:
                localgame.current_player = next(localgame.turn_order)
                localgame.board.do_visual()

                # Get board state as flat vector for the AI
                inputs = list(localgame.board.board)
                net_plays = sess.run(results, feed_dict={x: [inputs]})
                max_value = max(net_plays[0])
                max_index = net_plays[0].tolist().index(max_value)
                print(net_plays, max_index)
                #play whatever the AI weights highest
                p1.move = max_index
                is_valid = localgame.current_player.play(localgame.board)
                if is_valid == False:
                  print("Invalid Move!")
                  break
                #spacial case, punish AI player for making an invalid move.
                history.append((inputs, max_index))
                localgame.winner = localgame.board.check_win()
            if localgame.winner:
                print("Game Over!, {} Wins!".format(localgame.current_player.name))
                if localgame.current_player.name == 'ai_player':
                  #take winning games and build training data
                  won_games.append(history)
            else:
                print("Game Over!, No one Wins!")

        print(len(won_games))
        for game in won_games:
            for move in game:
                inputvals.append(move[0])
                targetvals.append(vote_for(move[1]))
        print(len(won_games)/float(ITTERCOUNT))
        #print(won_games, "\n============\n", inputvals, "\n============\n", targetvals)


# play a bunch of games and remember the ones we won
def play_games():
    won_games=[]
    with tf.name_scope('play_game'):
        for _ in range(ITTERCOUNT):
            raw_input("Press Enter to continue...")
            #generate new game
                #Do a standard human random game.
            p1=Player('ai_player', 'AI', 1)
            p2=Player('random_player', 'random', 0)
            board=Board()
            localgame=TicTacGame(board, p1, p2)
            history=[]

            while localgame.winner==None:
                localgame.current_player = next(localgame.turn_order)
                localgame.board.do_visual()

                # Get board state as flat vector for the AI
                inputs = list(localgame.board.board)
                net_plays = sess.run(results, feed_dict={x: [inputs]})
                max_value = max(net_plays[0])
                max_index = net_plays[0].tolist().index(max_value)
                print(net_plays, max_index)
                #play whatever the AI weights highest
                p1.move = max_index
                is_valid = localgame.current_player.play(localgame.board)
                if is_valid == False:
                  print("Invalid Move!")
                  break
                #spacial case, punish AI player for making an invalid move.
                history.append((inputs, max_index))
                localgame.winner = localgame.board.check_win()
            if localgame.winner:
                print("Game Over!, {} Wins!".format(localgame.current_player.name))
                if localgame.current_player.name == 'ai_player':
                  #take winning games and build training data
                  won_games.append(history)
            else:
                print("Game Over!, No one Wins!")

        print("games won:",len(won_games))
        for game in won_games:
            for move in game:
                inputvals.append(move[0])
                targetvals.append(vote_for(move[1]))
        print("won percent:", (len(won_games)/float(ITTERCOUNT))*100)
        #print(won_games, "\n============\n", inputvals, "\n============\n", targetvals)

# play a bunch of games and remember the ones we won
def play_games_self():
    won_games=[]
    with tf.name_scope('play_game'):
        for _ in range(ITTERCOUNT):
            #generate new game
                #Do a standard human random game.
            p1=Player('ai_player1', 'AI', 1)
            p2=Player('ai_player2', 'AI', 0)
            board=Board()
            localgame=TicTacGame(board, p1, p2)
            history=[]

            while localgame.winner==None:
                localgame.current_player = next(localgame.turn_order)
                #localgame.board.do_visual()

                # Get board state as flat vector for the AI
                inputs = list(localgame.board.board)
                net_plays = sess.run(results, feed_dict={x: [inputs]})
                max_value = max(net_plays[0])
                max_index = net_plays[0].tolist().index(max_value)
                #print(net_plays, max_index)
                #play whatever the AI weights highest
                p1.move = max_index

                #allow second guess for not improving net
                is_valid=False
                while not is_valid:
                  max_value = max(net_plays[0])
                  p2.move = net_plays[0].tolist().index(max_value)
                  is_valid = localgame.board.is_valid_play(p2.move)
                  if is_valid == False:
                    del net_plays[0][p2.move]


                is_valid = localgame.current_player.play(localgame.board)
                if is_valid == False:
                  print("Invalid Move!")
                  break
                #spacial case, punish AI player for making an invalid move.
                history.append((inputs, max_index))
                localgame.winner = localgame.board.check_win()
            if localgame.winner:
                print("Game Over!, {} Wins!".format(localgame.current_player.name))
                if localgame.current_player.name == 'ai_player1':
                  #take winning games and build training data
                  won_games.append(history)
            else:
                print("Game Over!, No one Wins!")

        print("games won:",len(won_games))
        for game in won_games:
            for move in game:
                inputvals.append(move[0])
                targetvals.append(vote_for(move[1]))
        print("won percent:", (len(won_games)/float(ITTERCOUNT))*100)
        #print(won_games, "\n============\n", inputvals, "\n============\n", targetvals)

# Input Layer > Hidden Layer
with tf.name_scope('layer1'):
  # initialize with a little noise and since we're using ReLU, we give them
  # a slightly positive bias
  W_fc1 = tf.truncated_normal([NUM_INPUTS, NUM_HIDDEN], mean=0.5, stddev=0.707)
  W_fc1 = tf.Variable(W_fc1, name='W_fc1')

  b_fc1 = tf.truncated_normal([NUM_HIDDEN], mean=0.5, stddev=0.707)
  b_fc1 = tf.Variable(b_fc1, name='b_fc1')

  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  tf.summary.histogram('W_fc1_summary', W_fc1)
  tf.summary.histogram('b_fc1_summary', b_fc1)
  tf.summary.histogram('h_fc1_summary', h_fc1)

# Hidden Layer -> Output layer
with tf.name_scope('layer2'):
  W_fc2 = tf.truncated_normal([NUM_HIDDEN, NUM_OUTPUTS], mean=0.5, stddev=0.707)
  W_fc2 = tf.Variable(W_fc2, name='W_fc2')

  b_fc2 = tf.truncated_normal([NUM_OUTPUTS], mean=0.5, stddev=0.707)
  b_fc2 = tf.Variable(b_fc2, name='b_fc2')

  y = tf.matmul(h_fc1, W_fc2) + b_fc2

  results = tf.sigmoid(y, name='results')

  tf.summary.histogram('W_fc2_summary', W_fc2)
  tf.summary.histogram('b_fc2_summary', b_fc2)
  tf.summary.histogram('y_summary', y)

# Error Calculation for Feedback
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
  tf.summary.scalar('cross_entropy', cross_entropy)

# Training step (backprop)
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.1, momentum=0.5).minimize(cross_entropy)

if write_for_tensorboard == 1:
  merged_summary = tf.summary.merge_all()
  print("Writing for TensorBoard to file %s"%(tensorboard_file))
  writer = tf.summary.FileWriter(tensorboard_file)
  writer.add_graph(sess.graph)

if do_training == 1:
  sess.run(tf.global_variables_initializer())

  #itercount games with 9 moves each
  memory = ITTERCOUNT*9
  #play some games then study what we won, repeat

  #play a random game to give the machine something to study.

  #remember some stuff but let trash fall out
  inputvals = inputvals[-memory:]
  targetvals = targetvals[-memory:]
  print(len(targetvals))
  play_games_random()

  for i in range(500):
      if i%100 == 0:
        train_error = cross_entropy.eval(feed_dict={x: inputvals, y_:targetvals})
        print("step %d, training error  %g"%(i, train_error))

      if write_for_tensorboard == 1 and i%5 == 0:
        s = sess.run(merged_summary, feed_dict={x: inputvals, y_:targetvals})
        writer.add_summary(s, i)

      sess.run(train_step, feed_dict={x: inputvals, y_: targetvals})

  for _ in range(30):

    #remember some stuff but let trash fall out
    inputvals = inputvals[-memory:]
    targetvals = targetvals[-memory:]
    print('memory depth:', len(targetvals))
    play_games()

    for i in range(500):
        if i%100 == 0:
          train_error = cross_entropy.eval(feed_dict={x: inputvals, y_:targetvals})
          print("step %d, training error  %g"%(i, train_error))

        if write_for_tensorboard == 1 and i%5 == 0:
          s = sess.run(merged_summary, feed_dict={x: inputvals, y_:targetvals})
          writer.add_summary(s, i)

        sess.run(train_step, feed_dict={x: inputvals, y_: targetvals})
    #raw_input("Press Enter to continue...")

  # for _ in range(40):

  #   #remember some stuff but let trash fall out
  #   inputvals = inputvals[-memory:]
  #   targetvals = targetvals[-memory:]
  #   print('memory depth:', len(targetvals))
  #   play_games_self()

  #   for i in range(500):
  #       if i%100 == 0:
  #         train_error = cross_entropy.eval(feed_dict={x: inputvals, y_:targetvals})
  #         print("step %d, training error  %g"%(i, train_error))

  #       if write_for_tensorboard == 1 and i%5 == 0:
  #         s = sess.run(merged_summary, feed_dict={x: inputvals, y_:targetvals})
  #         writer.add_summary(s, i)

  #       sess.run(train_step, feed_dict={x: inputvals, y_: targetvals})

  #   if save_trained == 1:
  #       print("Saving neural network to %s.*"%(save_file))
  #       saver = tf.train.Saver()
  #       saver.save(sess, save_file)
  #   #raw_input("Press Enter to continue...")

else: # if we're not training then we must be loading from file to play

  print("Loading neural network from %s"%(save_file))
  saver = tf.train.Saver()
  saver.restore(sess, save_file)
  # Note: the restore both loads and initializes the variables
  play_human()