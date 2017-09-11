# coding=utf-8
import numpy as np
from layers import *


class FullyConnectNet(object):
	"""docstring for ClassName"""
	def __init__(self, layers=[], input_dim=10, reg=0.5, mode='train'):

		# 此处接受layers(list or tuple)由用户自己定义多少层，每层多少个神经元

		self.layers = layers
		self.input_dim = input_dim
		self.reg = reg
		self.mode = mode

		# 初始化各层参数

		self.params = {}

		for idx, neurons in enumerate(layers):

			if idx == 0:

				self.params['W1'] = np.random.randn(input_dim, neurons)

			else:

				self.params['W%d'%(idx + 1)] = np.random.randn(layers[idx - 1], neurons)

			self.params['b%d'%(idx + 1)] = np.random.rand(neurons)


	# 这里最后一层用softmax计算loss, 在参数为self.params时， current mini-batch计算出的损失和梯度

	def loss(self, mini_batch_x, mini_batch_y=None):

		# 下面逐层计算前向传播及反向传播
		layer_counts = len(self.layers)

		layer_out = None
		layer_cache = None
		cache = []

		regulation = 0

		for layer in np.arange(layer_counts):

			# 如果是第一层
			if layer == 0:

				layer_out, layer_cache = affine_relu_forward(mini_batch_x, self.params['W1'], self.params['b1'])

			# 如果是最后一层
			elif layer == layer_counts - 1:

				layer_out, layer_cache = affine_forward(layer_out, self.params['W%d'%(layer + 1)], self.params['b%d'%(layer + 1)])
			else:

				layer_out, layer_cache = affine_relu_forward(layer_out, self.params['W%d'%(layer + 1)], self.params['b%d'%(layer + 1)])

			cache.append(layer_cache)

		# 如果是test mode，
		# if self.mode == 'test':
		if mini_batch_y is None:
			return layer_out

		# 计算loss并且开始反向传播

		dW = None

		db = None

		grads = {}

		loss, dlayer_out = softmax_loss(layer_out, mini_batch_y)

		for layer in range(layer_counts):

			regulation += np.sum(self.params['W%d'%(layer + 1)] ** 2)

		loss += 0.5 * self.reg * regulation

		for layer in reversed(np.arange(layer_counts)):

			if layer == layer_counts - 1:

				dlayer_out, dW, db = affine_backward(dlayer_out, cache[layer])

			else:

				dlayer_out, dW, db = affine_relu_backward(dlayer_out, cache[layer])

			grads['W%d'%(layer + 1)] = dW + self.reg * self.params['W%d'%(layer + 1)]

			grads['b%d'%(layer + 1)] = db + self.reg * self.params['b%d'%(layer + 1)]

		return loss, grads

	def predict(self, X):

		layer_counts = len(self.layers)

		layer_out = None

		for layer in np.arange(layer_counts):

			# 如果是第一层
			if layer == 0:

				layer_out, _ = affine_relu_forward(X, self.params['W1'], self.params['b1'])

			# 如果是最后一层
			elif layer == layer_counts - 1:

				layer_out, _ = affine_forward(layer_out, self.params['W%d'%(layer + 1)], self.params['b%d'%(layer + 1)])
			else:

				layer_out, _ = affine_relu_forward(layer_out, self.params['W%d'%(layer + 1)], self.params['b%d'%(layer + 1)])

		return np.argmax(layer_out, axis=1)
		