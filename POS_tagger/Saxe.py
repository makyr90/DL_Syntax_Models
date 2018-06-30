import numpy as np

class Orthogonal:

#https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py

    def __init__(self, gain=1.0, alpha = 0.1):
        if gain == 'relu':
            gain = np.sqrt(2)
        elif gain == 'leaky_relu':
            gain = np.sqrt(2/(1+alpha**2))


        self.gain = gain

    def __call__(self, shape):

        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        X = np.random.random((shape[0],shape[1]))
        u, _, v = np.linalg.svd(X, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q
