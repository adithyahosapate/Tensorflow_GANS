import numpy as np 
import tensorflow as tf


class DC_GAN:
	def __init__():
		pass

	def discriminator(x_image,reuse=True):
		if(reuse):
			tf.get_variable_scope().reuse_variables()
		d_weight1=tf.get_variable('d_w1',[filter_size,filter_size,1,32],
			initializer=tf.truncated_normal_initializer(stddev=0.02))	
		d_bias1=tf.get_variable('d_b1',[32],initializer=tf.constant_initializer(0))

		d1=tf.nn.conv2d(input=x_image,filter=d_weight1,strides=[1,1,1,1],padding='SAME')
		d1=tf.nn.relu(d1+d_bias1)


		d1=tf.nn.avg_pool(d1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		d_weight2=tf.get_variable('d_w2',[filter_size,filter_size,32,64],
		initializer=tf.truncated_normal_initializer(stddev=0.02))	
		d_bias2=tf.get_variable('d_b2',[64],initializer=tf.constant_initializer(0))

		d2=tf.nn.conv2d(input=x_image,filter=d_weight2,strides=[1,1,1,1],padding='SAME')
		d2=tf.nn.relu(d2+d_bias2)


		d2=tf.nn.avg_pool(d2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

		d_weight3=tf.get_variable('d_w3',[filter_size,filter_size,64,64],
		initializer=tf.truncated_normal_initializer(stddev=0.02))	
		d_bias3=tf.get_variable('d_b3',[64],initializer=tf.constant_initializer(0))

		d3=tf.nn.conv2d(input=x_image,filter=d_weight3,strides=[1,1,1,1],padding='SAME')
		d3=tf.nn.relu(d3+d_bias3)


		d3=tf.nn.avg_pool(d3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


		d_w4 = tf.get_variable('d_w4', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [1024], initializer=tf.constant_initializer(0))
		d4 = tf.reshape(d3, [-1, 7 * 7 * 64])
		d4 = tf.matmul(d3, d_w3)
		d4 = d4 + d_b4
		d4 = tf.nn.relu(d4)

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
		pass



