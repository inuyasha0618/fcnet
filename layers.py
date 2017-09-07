import numpy as np

def affine_forward(X, W, b):
	out = None
	# x.shape == (N, D1, D2, ...)
	# W.shape == (D1 * D2 * ..., M)
	# b.shape == (M, )

	# print(X.shape)

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


