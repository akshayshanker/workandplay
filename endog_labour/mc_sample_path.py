import quantecon as qe
import numpy as np



def mc_sample_path(P, psi=None, sample_size=1_000):

    # set up
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)

    # Convert each row of P into a cdf
    n = len(P)
    P_dist = [np.cumsum(P[i, :]) for i in range(n)]

    for i in range(len(P)):
        P_dist[i][-1] = 1

    # draw initial state, defaulting to 0
    if psi is not None:
        X_0 = qe.random.draw(np.cumsum(psi))
    else:
        X_0 = 0

    # simulate
    X[0] = X_0
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])

    return X