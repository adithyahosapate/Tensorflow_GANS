import tensorflow as tf
import numpy as np 
import datetime 
import matplotlib.pyplot as plt
#import esential libraries
import simple_GAN
from simple_GAN import simpleGAN

# Importing our Dataset - for this example, we will use MNIST
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


#Initialize our GAN class from our helper function
GAN=simpleGAN()




sess=tf.Session()

batch_size=50
z_dimensions=100
iterations=3000

x_placeholder=tf.placeholder("float32",shape=[None,28,28,1],name="placeholder_x")
with tf.variable_scope(tf.get_variable_scope()) as scope:
	Gz=simpleGAN.generator(batch_size,z_dimensions)
	Dx=simpleGAN.discriminator(x_placeholder)
	Dg = simpleGAN.discriminator(Gz, reuse=True)

g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
					logits=Dg,
					labels=tf.ones_like(Dg)))
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
					logits=Dx,
					labels=tf.fill([batch_size,1],0.9)))
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
					logits=Dg,
					labels=tf.zeros_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]
print(d_vars)
    # Next, we specify our two optimizers. In today’s era of deep learning, Adam seems to be the
    # best SGD optimizer as it utilizes adaptive learning rates and momentum. 
    # We call Adam's minimize function and also specify the variables that we want it to update
with tf.variable_scope(tf.get_variable_scope(), reuse=False):

	d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
	d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)

    # Train the generator    # Decreasing from 0.004 in GitHub version
with tf.variable_scope(tf.get_variable_scope(), reuse=False):
	g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)	

sess.run(tf.global_variables_initializer())



for iteration in range(iterations):
	real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

	_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                    {x_placeholder: real_image_batch})
	_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                    {x_placeholder: real_image_batch})
	_, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                                    {x_placeholder: real_image_batch})

	print(iteration)

	if iteration%10==0:
		print("Discriminator fake loss {}\nDiscriminator real loss {}\nGenerator loss {}\n".format(dLossFake,dLossReal,gLoss))
