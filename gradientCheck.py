import numpy as np

def numerical_gradient(f, X, dout, h=1e-5):

    dX = np.zeros_like(X)

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        index = it.multi_index

        curr_val = X[index]

        X[index] = curr_val - h

        # print(X)

        left = f(X).copy()

        X[index] = curr_val + h

        # print(X)

        right = f(X).copy()

        local_dcurr = (right - left) / (2 * h)

        # print(left)
        # print(right)

        dX[index] = np.sum(dout * local_dcurr)

        X[index] = curr_val

        it.iternext()

    return dX

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad