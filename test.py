import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2 as cv

from dataset import *
from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_test', 'mnist', 'model to run')
flags.DEFINE_string('checkpoint_path', 'model/model.ckpt', 'model checkpoint path')
flags.DEFINE_integer('retrieved_images', 7, 'number of retrieved images')
flags.DEFINE_string('module', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2', 'TF-Hub module to use')

# Helper function to plot image
def show_image(idxs, data):
  if type(idxs) != np.ndarray:
    idxs = np.array([idxs])
  fig = plt.figure()
  gs = gridspec.GridSpec(1,len(idxs))
  for i in range(len(idxs)):
    ax = fig.add_subplot(gs[0,i])
    ax.imshow(data[idxs[i]])
    ax.axis('off')
  plt.show()

# Calculate distances between images' features
def calculate_distances(features):
  num_feat = len(features)
  distances = np.zeros((num_feat, num_feat))
  for i in range(num_feat):
    search_feat = [features[i]]
    distances[i] = cdist(features, search_feat, 'cosine').ravel()
  return distances

# Calculate accuracy of a model
def calculate_accuracy(distances, labels):
  count = 0
  n = len(distances)
  for i in range(n):
    rank = np.argsort(distances[i])
    predict1 = labels[rank[0]]
    predict2 = labels[rank[1]]
    if labels[i] == predict1 and labels[i] == predict2:
      count += 1
  return count / n

if __name__ == "__main__":
  if FLAGS.model_test == 'mnist':
    dataset = MNISTDataset()
    img_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='img')
  elif FLAGS.model_test == 'custom':
    dataset = CustomDataset()
    img_placeholder = tf.placeholder(tf.float32, [None, 96, 96, 3], name='img')
  elif FLAGS.model_test == 'cifar10':
    dataset = Cifar10Dataset()
    img_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3], name='img')
  elif FLAGS.model_test == 'cifar100':
    dataset = Cifar100Dataset()
    img_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3], name='img')
  else:
    raise NotImplementedError("Model for %s is not implemented yet" % FLAGS.model_test)

  test_images = dataset.images_test
  labels_test = dataset.labels_test
  labels_name = dataset.labels_name
  len_test = len(test_images)

  # Create the siamese net feature extraction model
  net = model(img_placeholder, reuse=False)

  # Restore from checkpoint and calculate the features from all of train data
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, FLAGS.checkpoint_path)
    feats1 = sess.run(net, feed_dict={img_placeholder:test_images[:10000]})  

  # Create the CNN pre-trained feature extraction model
  print('Loading TF-Hub module...')
  module = hub.Module(FLAGS.module)
  height, width = hub.get_expected_image_size(module)

  print('Resizing images...')
  resized_images = np.zeros((len_test, height, width, 3))
  for i in range(len_test):
    img = cv.resize(test_images[i], dsize=(width, height), interpolation=cv.INTER_CUBIC)
    resized_images[i] = img
  print('Resized images :', resized_images.shape)

  print('Extracting features...')
  feats2 = module(resized_images)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feats2 = sess.run(feats2)

  print('Siamese net features :', feats1.shape)
  print('CNN pre-trained features :', feats2.shape)

  distances1 = calculate_distances(feats1)
  distances2 = calculate_distances(feats2)
  distances_avg = (distances1 + distances2) / 2

  accuracy1 = calculate_accuracy(distances1, labels_test)
  accuracy2 = calculate_accuracy(distances2, labels_test)
  accuracy_avg = calculate_accuracy(distances_avg, labels_test)

  print("Siamese net accuracy = ", accuracy1)
  print("CNN pre-trained accuracy = ", accuracy2)
  print("Average ensembling accuracy = ", accuracy_avg)

  # Searching for similar test images from trainset based on siamese feature
  # Generate new random test image
  idx = np.random.randint(0, len_test)

  # Show the test image
  print("Image id:", idx)
  print("Image label:", labels_name[labels_test[idx]])
  show_image(idx, test_images)

  # Run the test image through the network to get the test features
  saver = tf.train.Saver()
  search_feat = [feats1[idx]]

  # Calculate the cosine similarity and sort
  dist = cdist(feats1, search_feat, 'cosine')
  rank = np.argsort(dist.ravel())

  # Show the top n similar image from train data
  n = FLAGS.retrieved_images
  print("Retrieved ids:", rank[:n])
  print("Retrieved labels:", list(map(lambda x: labels_name[x], labels_test[rank[:n]])))
  show_image(rank[:n], test_images)
