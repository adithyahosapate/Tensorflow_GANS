import tensorflow as tf

class simpleGAN:

	def discriminator(x_image, reuse=False):
		if(reuse):
			tf.get_variable_scope().reuse_variables()

	# First convolutional and pool layers
	# These search for 32 different 5 x 5 pixel features
	#We’ll start off by passing the image through a convolutional layer. 
	#First, we create our weight and bias variables through tf.get_variable. 
	#Our first weight matrix (or filter) will be of size 5x5 and will have a output depth of 32. 
	#It will be randomly initialized from a normal distribution.
		d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
	#tf.constant_init generates tensors with constant values.
		d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))

		d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
	#add the bias
		d1 = d1 + d_b1
	#squash with nonlinearity (ReLU)
		d1 = tf.nn.relu(d1)
	##An average pooling layer performs down-sampling by dividing the input into 
	#rectangular pooling regions and computing the average of each region. 
	#It returns the averages for the pooling regions.
		d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#As with any convolutional neural network, this module is repeated, 
	# Second convolutional and pool layers
	# These search for 64 different 5 x 5 pixel features
		d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
		d2 = d2 + d_b2
		d2 = tf.nn.relu(d2)
		d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	 #and then followed by a series of fully connected layers. 
	# First fully connected layer
		d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
		d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
		d3 = tf.matmul(d3, d_w3)
		d3 = d3 + d_b3
		d3 = tf.nn.relu(d3)

	#The last fully-connected layer holds the output, such as the class scores.
	# Second fully connected layer
		d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

	#At the end of the network, we do a final matrix multiply and 
	#return the activation value. 
	#For those of you comfortable with CNNs, this is just a simple binary classifier. Nothing fancy.
	# Final layer
		d4 = tf.matmul(d3, d_w4) + d_b4
	# d4 dimensions: batch_size x 1

		return d4

	def generator(batch_size, z_dim):

		z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
	#first deconv block
		g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g1 = tf.matmul(z, g_w1) + g_b1
		g1 = tf.reshape(g1, [-1, 56, 56, 1])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
		g1 = tf.nn.relu(g1)

	# Generate 50 features
		g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
		g2 = g2 + g_b2
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
		g2 = tf.nn.relu(g2)
		g2 = tf.image.resize_images(g2, [56, 56])

	# Generate 25 features
		g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
		g3 = g3 + g_b3
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
		g3 = tf.nn.relu(g3)
		g3 = tf.image.resize_images(g3, [56, 56])

	# Final convolution with one output channel
		g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
		g4 = g4 + g_b4
		g4 = tf.sigmoid(g4)

	# No batch normalization at the final layer, but we do add
	# a sigmoid activator to make the generated images crisper.
	# Dimensions of g4: batch_size x 28 x 28 x 1

		return g4	
