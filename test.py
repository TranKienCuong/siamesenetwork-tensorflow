import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

from dataset import *

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
  print('Loading TF-Hub module...')
  module = hub.Module(FLAGS.module)
  height, width = hub.get_expected_image_size(module)

  resized_images = tf.image.resize_images(test_images, [height, width])

  print('Extracting features...')
  features = module(resized_images)

  with tf.variable_scope('CustomLayer'):
    dim = features.get_shape().as_list()[1]
    weight = tf.get_variable('weights', initializer=tf.truncated_normal((dim, dim)))
    bias = tf.get_variable('bias', initializer=tf.zeros((dim)))
    logits = tf.nn.xw_plus_b(features, weight, bias)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, FLAGS.checkpoint_path)
    feats = sess.run(logits)
  print('Features :', feats.shape)

  distances = calculate_distances(feats)
  accuracy = calculate_accuracy(distances, labels_test)
  print("Accuracy = ", accuracy)

  # Searching for similar test images from trainset based on siamese feature
  # Generate new random test image
  idx = np.random.randint(0, len_test)

  # Show the test image
  print("Image id:", idx)
  print("Image label:", labels_name[labels_test[idx]])
  show_image(idx, test_images)

  # Run the test image through the network to get the test features
  search_feat = [feats[idx]]

  # Calculate the cosine similarity and sort
  dist = cdist(feats, search_feat, 'cosine')
  rank = np.argsort(dist.ravel())

  # Show the top n similar image from train data
  n = FLAGS.retrieved_images
  print("Retrieved ids:", rank[:n])
  print("Retrieved labels:", list(map(lambda x: labels_name[x], labels_test[rank[:n]])))
  show_image(rank[:n], test_images)
