import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils import data_iterator
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

batch_size = 2
label_size = 2
dropout = 1
num_iter = 150


with tf.Graph().as_default():

	def load_data() :
		data =  np.random.randn(100,3)
		labels_1 = np.ones((len(data),1))
		labels_0 = np.zeros((len(data),1))
		labels = np.concatenate((labels_1, labels_0), axis = 1)
		indices = np.random.permutation(len(data))

		data = data[indices]
		labels = labels[indices]
		return data, labels
	

	def create_placeholder() :
		input_placeholder = tf.placeholder(tf.float32, 
	                              # shape=(self.config.batch_size, self.config.window_size),
	                              shape=None,
	                              name="input")
		labels_placeholder = tf.placeholder(tf.float32, 
	                              # shape=(self.config.batch_size, self.config.label_size),
	                              shape=None,
	                              name="labels")
		return input_placeholder, labels_placeholder


	def create_feed_dict(input_batch,label_batch=None):
	   
	    feed_dict = {
	        input_placeholder: input_batch,
	    }
	    if np.any(label_batch):
	        feed_dict[labels_placeholder] = label_batch
	    ### END YOUR CODE
	    return feed_dict

	def pred_train(x):
		"x shape : 1,3"
		"w shape : 3,3"
		"x w shape : 1,3"
		"b shape : 1,3"
		w = tf.get_variable("w", initializer = tf.random_normal([3,3]))
		b = tf.get_variable("b1", (3,), initializer = tf.constant_initializer(0.0))
	
		h1 = tf.nn.relu(tf.matmul(x,w)) 
		dropped1 = tf.nn.dropout(h1, dropout)

		U = tf.get_variable("U", initializer = tf.random_uniform([3,2], -1.0, 1.0))
		b2 = tf.get_variable("b2", (2,), initializer = tf.constant_initializer(0.0))

		h2 = tf.matmul(h1,U) + b2
		dropped2 = tf.nn.dropout(h2,dropout)
		
		out = dropped2

		return out


	def cross_entropy_loss(pred) :
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,labels_placeholder, name = 'xentropy')
		cross_entropy = tf.reduce_mean(cross_entropy)
		return cross_entropy

	def add_training_op(loss) :
		step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(
     	 0.2,   # Base learning rate.
      	step,  # Current index into the dataset.
      	1,     # Decay step.
      	1   # Decay rate
      	)
		train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  		train_op = train_op.minimize(loss, global_step = step)
  		return train_op



	with tf.Session() as sess :
		
		#load data
		input_data, input_labels = load_data()
		
		#create placeholders
		input_placeholder, labels_placeholder = create_placeholder()

		#create feed
 		feed = create_feed_dict(input_data, input_labels)

 		#create model
 		y = pred_train(input_placeholder)
 		print y

 		#define loss
		loss = cross_entropy_loss(y)

		#define training
		train_op = add_training_op(loss)
		#predictions
		predictions = tf.nn.softmax(y)
		one_hot_prediction = tf.argmax(predictions, 1)
		correct_prediction = tf.equal(tf.argmax(labels_placeholder, 1), one_hot_prediction)
		correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

		#init variables
		sess.run(tf.initialize_all_variables())
		losses = []
		iteration = 0
		for step in range (num_iter) :
			total_correct_examples = 0

			perte, total_correct, unused  = sess.run([loss, correct_predictions, train_op], feed_dict = feed )
			losses.append(perte)
			print "iteration", iteration, "loss", perte, "accuracy", total_correct / float(len(input_data))
			iteration += 1
		plt.plot(losses)
		plt.show()

