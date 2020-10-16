# Parzen Windows Method
import numpy as np

"""
    This method is computationally expensive, so it isn't applicable with real problems.
    Please, use more efficient libraries to compute Parzen Windows probability estimation (this is just a proof of concept).
"""


class GaussianFN():
    """
    Receives the precomputed parameters and use them to compute the results when requested
        data -> data points (array like)
        win  -> window of each dimension (bigger-->smoother distribution)
    """

    def __init__(self, data, win):
        self.data = data
        self.n = data.shape[1]  # number of dimension
        self.win = win  # window of each dimension
        self.denominator = (np.pi ** (self.n/2)) * \
            self.data.shape[0] * np.prod(self.win)

    """
    Receives a point (scalar or array like) and returns the value of the 
    """

    def __call__(self, x):
        x = np.array(x)
        p = 0
        for point in self.data:
            numerator = np.exp(-sum((x-point)**2 / self.win))
            p += numerator/self.denominator
        return p


class PWestimator():
    """
    Preprocess some parameters and stores them a new class (GaussianFN), which is returned.
    This latter class behaves as a function that map a point to the estimated probability

    estimation ::= sum_of(gaussian_kernel_i)/|samples|      , with every kernel being a probability distribution centered on a datapoint
    """

    def __init__(self):
        pass

    """
    Receives the data, preprocess them and returns a class capable of computing the estimated probability distribution
        data -> points
        win  -> window size for each dimension (default 1.. even if some heuristics could be used)
    """

    def fit(self, data, win=None):
        data = np.array(data)

        # even if the point is 1d, is seen as an array (np.reshape(1,*)?)
        if (len(data.shape) == 1):
            data = np.array([[x] for x in data])

        # window size of each dimension (similar to variance of a Normal)
        win = win or np.ones(data.shape[1])
        return GaussianFN(data, win)


# test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    data = [2, 3, 6, 4, 3, 4, 3]
    STEP_SIZE = 0.2

    # prabability distribution: dist(x) = probability density at point x
    dist = PWestimator().fit(data)
    if (dist):
        # points where to evaluate dist()
        X = np.arange(start=min(data)+1, stop=max(data)+1, step=STEP_SIZE)
        Y = []
        for x in X:
            pr = dist(x)
            Y.append(pr)

        # show plot density (not sure if plt works with higher dimansions...)
        plt.plot(X, Y)
        plt.show()
