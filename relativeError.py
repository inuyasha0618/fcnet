import numpy as np

def relative_error(A, B):
    abs = np.abs(A - B) / (np.maximum(1e-8, np.abs(A) + np.abs(B)))
    return np.max(abs)
