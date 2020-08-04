'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the solar database

tensorflow 1.0.1
'''

from __future__ import print_function
from tensorflow.contrib.rnn.python.ops import core_rnn as rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell
import tensorflow as tf
import solar_dataset
import tpr_fpr

solartrain,solartest = solar_dataset.read_data_sets(r'./',['train210.txt','trainlable210.txt','test210.txt','testlable210.txt'])
'''
To classify images using a reccurent neural network, we consider every image
row as a sequence of pixels. Because solar image shape is 120*120px, we will then
handle 120 sequences of 120 steps for every sample.
'''
with tf.device('/cpu:0'):
    # Parameters
    learning_rate = 0.0003  # learning_rate is varible
    training_iters = 80000
    batch_size = 100
    display_step = 10

    # Network Parameters
    n_input = 120 # solar data input (img shape: 120*120)
    n_steps = 120 # timesteps
    n_hidden = 360 # hidden layer num of features
    n_classes = 3 # solar total classes (0-no burst,1-burst,2-calibration)

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    ###########
    '''
	weights1 = {
	'out': tf.Variable(tf.random_normal([n_hidden,1]))
	}
	weights2 = {
	'out': tf.Variable(tf.random_normal([n_steps,n_classes]))
	}
	biases2 = {
	'out': tf.Variable(tf.random_normal([n_classes]))
	}
	'''
    def RNN(x, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Permuting batch_size and n_steps
		x = tf.transpose(x, [2, 0, 1])
		# Reshaping to (n_steps*batch_size, n_input)
		x = tf.reshape(x, [-1, n_input])
		# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
		x = tf.split(axis=0, num_or_size_splits=n_steps, value=x)

		# Define a lstm cell with tensorflow
		lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

		# Get lstm cell output
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		########### outputs==[n_steps,batch_size,n_hidden]
		#outputs1 = tf.reshape(outputs, [-1,n_hidden])
		#outputs2 = tf.matmul(outputs1, weights1['out'])
		#outputs3 = tf.reshape(outputs2, [-1,batch_size])
		#outputs4 = tf.matmul(outputs3, weights2['out'],transpose_a=True) + biases2['out']#'output1'is a 1-D array [n_hidden]
		#return outputs4
		# return tf.matmul(outputs4, weights['out']) + biases['out']

		# Linear activation, using rnn inner loop last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    argmax = tpr_fpr.tpr_fpr_argmax(pred, y)
	# Initializing the variables
    init = tf.initialize_all_variables()
# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step*batch_size < training_iters:
        batch_x, batch_y =solartrain.next_batch(batch_size)
        # Reshape data to get 120 seq of 120 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 960 solar test images

    i=1
    start =0
    end =0
    test_len = batch_size
    initflag=1
    classnumber = 3
    while i*test_len< 6416:
        end = i*test_len
        start = (i-1)*test_len
        test_data = solartest.images[start:end].reshape((-1, n_steps, n_input))
        test_label = solartest.labels[start:end]
        i = i+1
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        if initflag:
            tpr_fpr.tpr_fpr_init(classnumber)
            initflag=0
        argmaxlist = sess.run(argmax, feed_dict={x: test_data, y: test_label})
        tpr_fpr.tpr_fpr_statistics(argmaxlist)
    TPR,FPR = tpr_fpr.tpr_fpr_compute()
    print( "calibration",  "TPR2 = %s, FPR2 = %s" % (TPR[2],FPR[2]))
    print(    "burst",     "TPR1 = %s, FPR1 = %s" % (TPR[1],FPR[1]))
    print(   "no_burst",   "TPR0 = %s, FPR0 = %s" % (TPR[0],FPR[0]))
