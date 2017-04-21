from __future__ import print_function


#Supress TF Spam
import os
from tictac import TicTac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Some globals :/ dont hate.

do_training = 1 # 1 = do the training, 0 = load from file and just run it
save_trained = 0 # 1 = save to file after training, 0 = don't save
# Change the following to where you want the network to be saved to.
# Make sure to create the directory structure.
save_file = './tflogdir/model.ckpt'
write_for_tensorboard = 1 # 1 = write info for TensorBoard, 0 = don't
# Change the following to where you want the info to be saved.
# Make sure to create the directory structure.
tensorboard_file = './tflogdir/tftictac/1'

NUM_INPUTS = 18
NUM_HIDDEN = 64
NUM_OUTPUTS = 9
ITTERCOUNT = 400

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
def play_games():
    won_games=[]
    with tf.name_scope('play_game'):
        for _ in range(ITTERCOUNT):
            #generate new game
            game = TicTac(net=True, rand=True)

            # Get board state as flat vector
            inputs = list(game.boards[0])
            inputs.extend(game.boards[1])

            res = sess.run(results, feed_dict={x: [inputs]})
            
            while game.winner==False:
                game.visual()
                game.doturn(netvals=res[0])
                game.winner = game.check_win()
            if game.history[0]=='X':
                #learn the last two moves
                won_games.append(game.history[1:])
        #take winning games and build training data
        print(len(won_games))
        for game in won_games:
            for move in game:
                inputvals.append(move[0])
                targetvals.append(vote_for(move[1]))
        print(len(won_games)/float(ITTERCOUNT))
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
  for _ in range(100):

    #remember some stuff but let trash fall out
    inputvals = inputvals[-memory:]
    targetvals = targetvals[-memory:]
    print(len(targetvals))
    play_games()

    for i in range(500):
        if i%100 == 0:
          train_error = cross_entropy.eval(feed_dict={x: inputvals, y_:targetvals})
          print("step %d, training error  %g"%(i, train_error))
          if train_error < 0.005:
            break

        if write_for_tensorboard == 1 and i%5 == 0:
          s = sess.run(merged_summary, feed_dict={x: inputvals, y_:targetvals})
          writer.add_summary(s, i)

        sess.run(train_step, feed_dict={x: inputvals, y_: targetvals})

    if save_trained == 1:
        print("Saving neural network to %s.*"%(save_file))
        saver = tf.train.Saver()
        saver.save(sess, save_file)

else: # if we're not training then we must be loading from file to play

  print("Loading neural network from %s"%(save_file))
  saver = tf.train.Saver()
  saver.restore(sess, save_file)
  # Note: the restore both loads and initializes the variables
  play_a_human()