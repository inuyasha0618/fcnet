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