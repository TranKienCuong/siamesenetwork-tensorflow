import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import glob
import os.path as path

from scipy.spatial.distance import cdist
from matplotlib import gridspec
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', 'model/model.ckpt', 'model checkpoint path')
flags.DEFINE_integer('retrieved_images', 7, 'number of retrieved images')
flags.DEFINE_string('module', 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2', 'TF-Hub module to use')
flags.DEFINE_boolean('debug', False, 'show wrong images or not')

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

def calculate_accuracy(distances, labels, group_labels, images, group_images):
  count = 0
  n = len(distances)
  for i in range(n):
    rank = np.argsort(distances[i])
    predict = group_labels[rank[0]]
    if labels[i] == predict:
      count += 1
    elif FLAGS.debug:
      show_image(i, images)
      show_image(rank[:10], group_images)
  return count / n

def calculate_accuracy_by_groups(distances, groups, labels, images):
  group_ids = np.unique(groups)
  accuracy = []
  for group_id in group_ids:
    group_distances = np.transpose(distances[np.where(groups == group_id)])
    group_labels = labels[np.where(groups == group_id)]
    group_images = images[np.where(groups == group_id)]
    accuracy.append(calculate_accuracy(group_distances, labels, group_labels, images, group_images))
  return np.mean(np.array(accuracy))

if __name__ == "__main__":
  images = []
  screens = []
  devices = []
  labels = []

   # Create the siamese net feature extraction model
  print(f'Loading TF-Hub module {FLAGS.module}...')
  module = hub.Module(FLAGS.module)
  height, width = hub.get_expected_image_size(module)
  print('Expected width height:', width, height)

  print('Resizing images...')
  for img_path in glob.glob("images/*_*_*.jpg"):
    img_name = path.splitext(path.basename(img_path))[0]
    screen, device, label = np.str.split(img_name, '_')
    img = Image.open(img_path)
    img = np.array(img.convert('RGB')) / 255.0
    img = tf.image.resize_image_with_pad(img, height, width)
    images.append(img)
    screens.append(screen)
    devices.append(device)
    labels.append(label)

  screens = np.array(screens)
  devices = np.array(devices)
  labels = np.array(labels)
  images = np.array(tf.Session().run(images))
  print('Resized images:', images.shape)

  len_test = len(images)
  # resized_images = tf.image.resize_images(test_images, [height, width])

  print(f'Extracting features from model {FLAGS.checkpoint_path}...')
  features = module(images)

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

  screen_ids = np.unique(screens)
  accuracy = []
  for screen_id in screen_ids:
    screen_indices = np.where(screens == screen_id)
    screen_feats = feats[screen_indices]
    screen_labels = labels[screen_indices]
    screen_images = images[screen_indices]
    screen_devices = devices[screen_indices]
    device_ids = np.unique(screen_devices)
    screen_distances = calculate_distances(screen_feats)
    accuracy.append(calculate_accuracy_by_groups(screen_distances, screen_devices, screen_labels, screen_images))
  print("Accuracy = ", np.mean(np.array(accuracy)))
