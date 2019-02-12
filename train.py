from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2 as cv

from dataset import *
from model import *
from scipy.spatial.distance import cdist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_float('margin', 0.5, 'Margin for contrastive loss')
flags.DEFINE_string('model_train', 'mnist', 'model to run')
flags.DEFINE_string('module', 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1', 'TF-Hub module to use')

if __name__ == "__main__":
	#setup dataset
	if FLAGS.model_train == 'mnist':
		dataset = MNISTDataset()
		colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	elif FLAGS.model_train == 'custom':
		dataset = CustomDataset()
		colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	elif FLAGS.model_train == 'cifar10':
		dataset = Cifar10Dataset()
		colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	elif FLAGS.model_train == 'cifar100':
		dataset = Cifar100Dataset()
		colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	else:
		raise NotImplementedError("Model for %s is not implemented yet" % FLAGS.model_train)

	module = hub.Module(FLAGS.module, trainable=True)
	height, width = hub.get_expected_image_size(module)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		dataset.images_train = sess.run(tf.image.resize_images(dataset.images_train, [height, width]))
		dataset.images_test = sess.run(tf.image.resize_images(dataset.images_test, [height, width]))
	
	placeholder_shape = [None] + list(dataset.images_train.shape[1:])
	print("placeholder_shape", placeholder_shape)

	left = tf.placeholder(tf.float32, placeholder_shape, name='left')
	right = tf.placeholder(tf.float32, placeholder_shape, name='right')

	features_left = module(left)
	features_right = module(right)

	with tf.variable_scope('CustomLayer'):
		dim = features_left.get_shape().as_list()[1]
		weight = tf.get_variable('weights', initializer=tf.truncated_normal((dim, dim)))
		bias = tf.get_variable('bias', initializer=tf.zeros((dim)))
		logits_left = tf.nn.xw_plus_b(features_left, weight, bias)
		logits_right = tf.nn.xw_plus_b(features_right, weight, bias)

	var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')

	# Setup network
	next_batch = dataset.get_siamese_batch
	with tf.name_scope("similarity"):
		label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
		label_float = tf.to_float(label)
	left_output = logits_left
	right_output = logits_right
	print('Margin: ', FLAGS.margin)
	loss = contrastive_loss(left_output, right_output, label_float, FLAGS.margin)

	# Setup Optimizer
	global_step = tf.Variable(0, trainable=False)

	# starter_learning_rate = 0.0001
	# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
	# tf.scalar_summary('lr', learning_rate)
	# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step, var_list=var_list)
	# train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

	# Start Training
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#setup tensorboard	
		tf.summary.scalar('step', global_step)
		tf.summary.scalar('loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('train.log', sess.graph)

		#train iter
		for i in range(FLAGS.train_iter):
			batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size)

			_, l, summary_str = sess.run([train_step, loss, merged],
										 feed_dict={left:batch_left, right:batch_right, label: batch_similarity})
			
			writer.add_summary(summary_str, i)
			print("\r#%d - Loss"%i, l)
			
			if (i + 1) % FLAGS.step == 0:
				#generate test
				# TODO: create a test file and run per batch
				feat = sess.run(left_output, feed_dict={left:dataset.images_test})
				labels = dataset.labels_test
				count = 0
				n = dataset.images_test.shape[0]
				for j in range(n):
					search_feat = [feat[j]]
					dist = cdist(feat, search_feat, 'cosine')
					rank = np.argsort(dist.ravel())
					predict1 = labels[rank[0]]
					predict2 = labels[rank[1]]
					if labels[j] == predict1 and labels[j] == predict2:
						count += 1
				print("Accuracy = ", count / n)
					
				# plot result
				f = plt.figure(figsize=(16,9))
				f.set_tight_layout(True)
				for j in range(colors.__len__()):
				    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(), '.', c=colors[j], alpha=0.8)
				plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
				plt.savefig('img/%d.jpg' % (i + 1))

		saver.save(sess, "model/model.ckpt")
