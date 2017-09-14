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

		d3=tf.contrib.layers.flatten(d3)
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

		#credits:Adit Deshpande
		g_dim = 64 #Number of filters of first layer of generator 
        c_dim = 1 #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
        s = 28 #Output size of the image
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #We want to slowly upscale the image, so these values will help
                                                                  #make that change gradual.

        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)
        output1_shape = [batch_size, s8, s8, g_dim*4]

        W_conv1 = tf.get_variable('g_wconv1', 
        	[5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
        	initializer=truncated_normal_initializer(stddev=0.02))
		
		b_conv1=tf.get_variable('g_wb1', [output1_shape[-1]],initializer=constant_initializer(1))

		H_conv1=tf.nn.conv2d_transpose(input=h0,filters=W_conv1,output_shape=output1_shape,strides=[1,2,2,1],padding="SAME")

		H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")

		H_conv1=tf.nn.relu(H_conv1+b_conv1)
		output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]

		W_conv2=tf.get_variable('g_wconv2',
			[5,5,output2_shape[-1],int(H_conv1.get_shape()[-1])],
			initializer=truncated_normal_initializer(stddev=0.02))

		b_conv2=tf.get_variable('g_wb2',[output2_shape[-1]],initializer=constant_initializer(1))

		H_conv2=tf.nn.conv2d_transpose(input=H_conv1,filters=W_conv2,output_shape=output2_shape,strides=[1,2,2,1],padding='SAME')

		H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")

		H_conv2=tf.nn.relu(H_conv2+b_conv2)

		output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        #Dimensions of H_conv3 = batch_size x 12 x 12 x 64

        #Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID')
        H_conv4 = tf.nn.tanh(H_conv4)
        #Dimensions of H_conv4 = batch_size x 28 x 28 x 1

    	return H_conv4









