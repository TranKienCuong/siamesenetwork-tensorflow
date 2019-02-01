import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_test', 'mnist', 'model to run')
flags.DEFINE_string('checkpoint_path', 'model/model.ckpt', 'model checkpoint path')
flags.DEFINE_integer('retrieved_images', 7, 'number of retrieved images')

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
    feats = sess.run(net, feed_dict={img_placeholder:test_images[:10000]})                

  # Calculate accuracy of the model
  count = 0
  n = dataset.images_test.shape[0]
  for j in range(n):
    search_feat = [feats[j]]
    dist = cdist(feats, search_feat, 'cosine')
    rank = np.argsort(dist.ravel())
    predict1 = labels_test[rank[0]]
    predict2 = labels_test[rank[1]]
    if labels_test[j] == predict1 and labels_test[j] == predict2:
      count += 1
  print("Accuracy = ", count / n)

  # Searching for similar test images from trainset based on siamese feature
  # Generate new random test image
  idx = np.random.randint(0, len_test)
  im = test_images[idx]

  # Show the test image
  print("Image id:", idx)
  print("Image label:", labels_name[labels_test[idx]])
  show_image(idx, test_images)

  # Run the test image through the network to get the test features
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, FLAGS.checkpoint_path)
    search_feat = sess.run(net, feed_dict={img_placeholder:[im]})

  # Calculate the cosine similarity and sort
  dist = cdist(feats, search_feat, 'cosine')
  rank = np.argsort(dist.ravel())

  # Show the top n similar image from train data
  n = FLAGS.retrieved_images
  print("Retrieved ids:", rank[:n])
  print("Retrieved labels:", list(map(lambda x: labels_name[x], labels_test[rank[:n]])))
  show_image(rank[:n], test_images)
