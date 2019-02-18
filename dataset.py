from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'mnist', 'model to run')

class Dataset(object):
	images_train = np.array([])
	images_test = np.array([])
	labels_train = np.array([])
	labels_test = np.array([])
	unique_train_label = np.array([])
	labels_name = np.array([])
	map_train_label_indices = dict()
	channels = 1

	def _get_siamese_similar_pair(self):
		label =np.random.choice(self.unique_train_label)
		l, r = np.random.choice(self.map_train_label_indices[label], 2, replace=False)
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):
		label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
		l = np.random.choice(self.map_train_label_indices[label_l])
		r = np.random.choice(self.map_train_label_indices[label_r])
		return l, r, 0

	def _get_siamese_pair(self):
		if np.random.random() < 0.5:
			return self._get_siamese_similar_pair()
		else:
			return self._get_siamese_dissimilar_pair()

	def get_siamese_batch(self, n):
		idxs_left, idxs_right, labels = [], [], []
		for _ in range(n):
			l, r, x = self._get_siamese_pair()
			idxs_left.append(l)
			idxs_right.append(r)
			labels.append(x)
		return self.images_train[idxs_left,:], self.images_train[idxs_right, :], np.expand_dims(labels, axis=1)

class MNISTDataset(Dataset):
	def __init__(self):
		print("===Loading MNIST Dataset===")
		self.channels = 1
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = mnist.load_data()
		self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
		self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)
		self.unique_train_label = np.unique(self.labels_train)
		self.labels_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

class CustomDataset(Dataset):
	def __init__(self):
		print("===Loading Custom Dataset===")
		self.channels = 3

		files = [
			'./dataset/train-labels-idx1-ubyte.gz',
			'./dataset/train-images-idx3-ubyte.gz',
			'./dataset/test-labels-idx1-ubyte.gz',
			'./dataset/test-images-idx3-ubyte.gz'
		]

		width = 96
		height = 96

		with gzip.open(files[0], 'rb') as lbpath:
			self.labels_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

		with gzip.open(files[1], 'rb') as imgpath:
			self.images_train = np.frombuffer(
			imgpath.read(), np.uint8, offset=16).reshape(len(self.labels_train), width, height, self.channels)

		with gzip.open(files[2], 'rb') as lbpath:
			self.labels_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

		with gzip.open(files[3], 'rb') as imgpath:
			self.images_test = np.frombuffer(
			imgpath.read(), np.uint8, offset=16).reshape(len(self.labels_test), width, height, self.channels)

		self.images_train = self.images_train / 255.0
		self.images_test = self.images_test / 255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)
		self.unique_train_label = np.unique(self.labels_train)
		self.labels_name = ["align", "balloon", "dislike", "shoe", "truck", "remove", "search", "send", "settings", "share",
			"chair", "door", "graph", "remote", "speaker"]
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

class Cifar10Dataset(Dataset):
	def __init__(self):
		print("===Loading Cifar10 Dataset===")
		self.channels = 3
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = cifar10.load_data()
		self.images_train = self.images_train / 255.0
		self.images_test = self.images_test / 255.0
		self.unique_train_label = np.unique(self.labels_train)
		self.labels_test = np.squeeze(self.labels_test, axis=1)
		self.labels_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
			"ship", "truck", "account", "add", "airplane", "back", "battery", "bell", "bluetooth",
			"calculator", "camera", "clock", "close", "cut", "download", "edit", "file", "fingerprint",
			"fire", "flag", "fullscreen", "gift", "hamburger", "headphones", "heart", "home", "info",
			"like", "location", "lock", "mail", "map", "message", "microphone", "music", "pause", "pin",
			"play", "question", "refresh", "save", "star", "tag", "trophy", "unlock", "upload", "volume"]
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

class Cifar100Dataset(Dataset):
	def __init__(self):
		print("===Loading Cifar100 Dataset===")
		self.channels = 3
		(self.images_train, self.labels_train), (self.images_test, self.labels_test) = cifar100.load_data()
		self.images_train = self.images_train / 255.0
		self.images_test = self.images_test / 255.0
		self.unique_train_label = np.unique(self.labels_train)
		self.labels_test = np.squeeze(self.labels_test, axis=1)
		self.labels_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
			'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
			'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
			'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
			'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard',
			'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
			'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
			'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray',
			'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
			'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
			'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
			'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	if (FLAGS.model == 'mnist'):
		dataset = MNISTDataset()
	elif (FLAGS.model == 'custom'):
		dataset = CustomDataset()
	elif (FLAGS.model == 'cifar10'):
		dataset = Cifar10Dataset()
	elif (FLAGS.model == 'cifar100'):
		dataset = Cifar100Dataset()
	else:
		raise NotImplementedError("Model for %s is not implemented yet" % FLAGS.model)

	batch_size = 4
	ls, rs, xs = dataset.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2)
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		if (dataset.channels == 1):
			l = np.squeeze(l, axis=2)
			r = np.squeeze(r, axis=2)
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("Max pixel value:", l.max())
		axarr[idx, 0].imshow(l)
		axarr[idx, 1].imshow(r)
	plt.show()