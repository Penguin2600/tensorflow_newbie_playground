tensorboard_file='./tboardfile'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sess = tf.InteractiveSession()

with tf.name_scope('adding_raw'):
	node1 = tf.constant(3.0, tf.float32)
	node2 = tf.constant(9.0)
	print(node1,node2)

	node3 = tf.add(node1, node2)
	result = sess.run(node3)
	print(result)

with tf.name_scope('adding_func'):
	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	adder_node = a + b 
	result = sess.run(adder_node, {a: 3, b:4.5})
	print(result)

with tf.name_scope('add_mult'):
	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	c = tf.placeholder(tf.float32)
	adder_node = a + b 
	add_mult = adder_node * c
	result = sess.run(add_mult, {a: 3, b:4.5, c:2})
	print(result)


with tf.name_scope('linear_model'):
	W = tf.Variable([.3], tf.float32)
	b = tf.Variable([-.3], tf.float32)
	x = tf.placeholder(tf.float32)
	linear_model = W * x + b
	init = tf.global_variables_initializer()
	sess.run(init)
	result=sess.run(linear_model, {x:[1,2,3,4]})
	print(result)

with tf.name_scope('mean_squared_error'):
	y = tf.placeholder(tf.float32)
	squared_deltas = tf.square(linear_model - y)
	loss = tf.reduce_sum(squared_deltas)
	error=sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})
	print(error)

with tf.name_scope('training'):
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)
	# training data
	x_train = [1,2,3,4]
	y_train = [0,-1,-2,-3]
	# training loop
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) # reset values to wrong
	for i in range(1000):
		sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
merged_summary = tf.summary.merge_all()
print("Writing for TensorBoard to file %s"%(tensorboard_file))
writer = tf.summary.FileWriter(tensorboard_file)
writer.add_graph(sess.graph)
writer.flush()
