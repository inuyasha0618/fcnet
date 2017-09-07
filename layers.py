def affine_forward(X, W, b):
	out = None
	# x.shape == (N, D)
	# W.shape == (D, M)
	# b.shape == (1, M)
	N, D = X.shape
	_, M = W.shape
	
	y = X.reshape(N, -1).dot(W) + b
	cache = (X, W, b)

	return out, cache


