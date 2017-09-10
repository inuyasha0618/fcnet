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

	N, D = X.shape

	max_score =  np.max(X, axis=1, keepdims=True)

	max_mask = X == max_score

	safe_X = X - max_score

	exp_X = np.exp(safe_X)

	exp_sum = np.sum(exp_X, axis=1, keepdims=True)

	exp_proportion = exp_X / exp_sum

	log_out = -np.log(exp_proportion)

	y_index_mask = np.zeros_like(X)

	y_index_mask[np.arange(N), y] = 1

	index_out = log_out * y_index_mask

	loss = np.sum(index_out) / N

	if y is None:
		return loss

	# back prop
	dsum = 1.0 / N

	dindex_out = dsum * np.ones_like(X)

	dlog_out = y_index_mask * dindex_out

	dexp_proportion = -1 / exp_proportion * dlog_out

	dexp1 = 1 / (exp_sum * np.ones_like(X)) * dexp_proportion

	dexp_sum = np.sum(-exp_X * dexp_proportion / (exp_sum ** 2), axis=1, keepdims=True)

	dexp2 = dexp_sum * np.ones_like(X)

	dexp_X = dexp1 + dexp2

	dsafe_X = exp_X * dexp_X

	dX1 = dsafe_X

	dmax = -np.sum(dsafe_X, axis=1, keepdims=True)

	dX2 = dmax * max_mask

	dX = dX1 + dX2

	return loss, dX


def softmax_loss_true(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
