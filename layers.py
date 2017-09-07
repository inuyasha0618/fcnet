def affine_forward(X, W, b):
	out = None
	# x.shape == (N, D1, D2, ...)
	# W.shape == (D1 * D2 * ..., M)
	# b.shape == (M, )

	N, D = X.shape
	_, M = W.shape
	
	y = X.reshape(N, -1).dot(W) + b

	cache = (X, W, b)

	return out, cache


def affine_backward(dout, cache):

	N, M = dout.shape

	X, W, b = cache

	dX = dout.dot(W.T).reshape(*X.shape)

	dW = np.dot(X.T, dout)

	db = np.sum(dout, axis=0)

	return dX, dW, db


