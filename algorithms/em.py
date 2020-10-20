# expectation maximization algorithm
#  estimate the probability density distribution as a sum of gaussian functions.
import numpy as np
import matplotlib.pyplot as plt


class EMalg():
    def __init__(self):
        pass

    def _get_initials(self, data, n_clusters):
        # probability of each gaussian
        p = np.ones(n_clusters) / n_clusters
        # center of each gaussian
        u = np.array([np.average(data, axis=0)]*n_clusters, copy=True) + \
            (np.random.rand(n_clusters, data.shape[1])-0.5)*2
        # covariance matrices of each gaussian
        E = np.array([np.identity(data.shape[1])] * n_clusters, copy=True)
        return p, u, E

    def _converged(self):  # needs to be updated
        return np.random.rand() > 0.95

    def fit(self, data, n_clusters=2, showPass=False):
        data = np.array(data)
        self.p, self.u, self.E = self._get_initials(data, n_clusters)
        if (showPass):
            self._show()

        # improve distributions at each step
        while (not self._converged()):
            pr_points = self._get_pr_points(data)  # expectation step
            self._update(pr_points, data)  # maximization step
            if (showPass):
                self._show()

    def _get_pr_points(self, data):
        # precompute var needed after
        p, u, E = self.p, self.u, self.E
        try:
            E_1 = np.linalg.inv(E)
        except Exception as e:
            print(f"ERROR: {e}\nmatrix:\n", E)
            exit(1)
        det_E = np.linalg.det(E)
        n_dim = data.shape[1]

        pr_points = []
        for x in data:  # for each point
            X_pr = []
            for i in range(len(E)):  # for each normal
                p1 = np.matmul((x-u[i]), E_1[i])
                p2 = np.matmul(p1, x-u[i])
                # compute prob. of point given normal
                pr = p[i]*np.exp(p2/-2)/(np.pi*2*det_E[i])**(n_dim/2)
                X_pr.append(pr)
            X_pr = np.array(X_pr)/sum(X_pr)
            pr_points.append(X_pr)

        return(np.array(pr_points))

    def _update(self, pr_points, data):
        # update the weight of the gaussians (Pr(gaussian_i))
        mc = np.sum(pr_points, axis=0)
        self.p = mc / sum(mc)
        n_dim = data.shape[1]

        # update central points of gaussians
        u = []
        for i in range(len(self.E)):
            ui = np.zeros(n_dim)
            for x, pr_x in zip(data, pr_points):
                ui += x * pr_x[i]
            ui /= mc[i]
            u.append(ui)
        self.u = u

        # update covariance matrix of gaussians
        E = []
        for i in range(len(self.E)):
            Ei = np.zeros([n_dim, n_dim])
            for x, pr_x in zip(data, pr_points):
                Ei += pr_x[i] * np.matmul((x-u[i]).reshape(n_dim, 1),
                                          (x-u[i]).reshape(1, n_dim))
            Ei /= mc[i]
            E.append(Ei)
        self.E = E

    def _show(self):
        print(self.u, f"\n_______________\n",
              self.p, f"\n_______{sum(self.p)}________\n",
              self.E, f"\n====||============")


if __name__ == '__main__':
    em = EMalg()
    data = np.array([
        [1, 3],
        [1, 5],
        [0.9, 4],
        [2, 15],
        [2, 14],
        [2.1, 14]
    ])
    em.fit(data)
