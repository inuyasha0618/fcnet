import numpy as np

def affine_forward(X, W, b):

	out = None

	N = X.shape[0]

	_, M = W.shape
	
	out = X.reshape(N, -1).dot(W) + b

	cache = (X, W, b)

	return out, cache


def affine_backward(dout, cache):

	N, M = dout.shape

	X, W, b = cache

	X_rows = X.reshape(X.shape[0], -1)

	dX = dout.dot(W.T).reshape(*X.shape)

	dW = np.dot(X_rows.T, dout)

	db = np.sum(dout, axis=0)

	return dX, dW, db

def relu_forward(X):

	cache = X > 0

	out = np.maximum(0, X)

	return out, cache

def relu_backward(dout, cache):

	pos_mask = cache

	local_derivative = np.zeros(pos_mask.shape)

	local_derivative[pos_mask] = 1

	dX = local_derivative * dout

	return dX

def affine_relu_forward(X, W, b):

	affine_out, affine_cache = affine_forward(X, W, b)

	out, relu_cache = relu_forward(affine_out)

	cache = (affine_cache, relu_cache)

	return out, cache

def affine_relu_backward(dout, cache):

	affine_cache, relu_cache = cache

	daffine_out = relu_backward(dout, relu_cache)

	dX, dW, db = affine_backward(daffine_out, affine_cache)

	return dX, dW, db

def softmax_loss(X, y):

	


