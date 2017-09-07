def numerical_gradient(f, X, dout, h=1e-5):

    dX = np.zeros_like(X)

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        index = it.multi_index

        curr_val = X[index]

        left = curr_val - h

        right = curr_val + h

        local_dcurr = (f(right) - f(left)) / (2 * h)

        dX[index] = np.sum(dout * local_dcurr)

        it.iternext()

    return dX