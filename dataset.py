from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import gzip
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class Dataset(object):
	images_train = np.array([])
	images_test = np.array([])
	labels_train = np.array([])
	labels_test = np.array([])
	unique_train_label = np.array([])
	map_train_label_indices = dict()

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
		# (self.images_train, self.labels_train), (self.images_test, self.labels_test) = mnist.load_data()

		files = [
			'./dataset/train-labels-idx1-ubyte.gz',
			'./dataset/train-images-idx3-ubyte.gz',
			'./dataset/test-labels-idx1-ubyte.gz',
			'./dataset/test-images-idx3-ubyte.gz'
		]

		width = 96
		height = 96
		channels = 3

		with gzip.open(files[0], 'rb') as lbpath:
			self.labels_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

		with gzip.open(files[1], 'rb') as imgpath:
			self.images_train = np.frombuffer(
			imgpath.read(), np.uint8, offset=16).reshape(len(self.labels_train), width, height, channels)

		with gzip.open(files[2], 'rb') as lbpath:
			self.labels_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

		with gzip.open(files[3], 'rb') as imgpath:
			self.images_test = np.frombuffer(
			imgpath.read(), np.uint8, offset=16).reshape(len(self.labels_test), width, height, channels)

		self.images_train = np.expand_dims(self.images_train, axis=3) / 255.0
		self.images_test = np.expand_dims(self.images_test, axis=3) / 255.0
		self.labels_train = np.expand_dims(self.labels_train, axis=1)
		self.unique_train_label = np.unique(self.labels_train)
		self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in self.unique_train_label}
		print("Images train :", self.images_train.shape)
		print("Labels train :", self.labels_train.shape)
		print("Images test  :", self.images_test.shape)
		print("Labels test  :", self.labels_test.shape)
		print("Unique label :", self.unique_train_label)
		# print("Map label indices:", self.map_train_label_indices)

if __name__ == "__main__":
	# Test if it can load the dataset properly or not. use the train.py to run the training
	a = MNISTDataset()
	batch_size = 4
	ls, rs, xs = a.get_siamese_batch(batch_size)
	f, axarr = plt.subplots(batch_size, 2)
	for idx, (l, r, x) in enumerate(zip(ls, rs, xs)):
		print("Row", idx, "Label:", "similar" if x else "dissimilar")
		print("max:", np.squeeze(l, axis=2).max())
		axarr[idx, 0].imshow(np.squeeze(l, axis=2))
		axarr[idx, 1].imshow(np.squeeze(r, axis=2))
	plt.show()
