import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorboard_file='tflogdir'
save_file='tflogdir/model.ckpt'

# training data
x_train = [1,2,3,4]
y_train = [-2,-4,-6,-8]

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sess = tf.InteractiveSession()

# The simplest model
with tf.name_scope('adding_raw'):
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(9.0)
    print(node1,node2)

    node3 = tf.add(node1, node2)
    result = sess.run(node3)
    print(result)

# Example of placeholders and arguements
with tf.name_scope('adding_func'):
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b 
    result = sess.run(adder_node, {a: 3, b:4.5})
    print(result)

# Example of a multi layer model
with tf.name_scope('add_mult'):
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    c = tf.placeholder(tf.float32)
    adder_node = a + b 
    add_mult = adder_node * c
    result = sess.run(add_mult, {a: 3, b:4.5, c:2})
    print(result)

# Example of a linear model, y=Wx+b 
with tf.name_scope('linear_model'):
    W = tf.Variable([.1], tf.float32)
    b = tf.Variable([.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    init = tf.global_variables_initializer()
    sess.run(init)
    result=sess.run(linear_model, {x:x_train})
    tf.summary.histogram('linear', linear_model)
    tf.summary.histogram('W', W)
    tf.summary.histogram('b', b)


    print(result)

# Simple loss calculations
with tf.name_scope('mean_squared_error'):
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    error=sess.run(loss, {x:x_train, y:y_train})
    tf.summary.scalar('loss', loss)
    print(error)

with tf.name_scope('training'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

# Setup for tensorboard logging
merged_summary = tf.summary.merge_all()
print("Writing for TensorBoard to file %s"%(tensorboard_file))
writer = tf.summary.FileWriter(tensorboard_file)
writer.add_graph(sess.graph)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
    s = sess.run(merged_summary, feed_dict={x: x_train, y:y_train})
    writer.add_summary(s, i)
    if i%100 == 0:
        train_error = sess.run(loss, {x:x_train, y:y_train})
        print("step %d, training error  %g"%(i, train_error))
        if train_error < 0.00001:
            break



# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

print("Saving neural network to %s.*"%(save_file))
saver = tf.train.Saver()
saver.save(sess, save_file,1)

writer.flush()