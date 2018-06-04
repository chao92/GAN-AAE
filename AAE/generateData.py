import numpy as np

N = 1000
def random_symmetric_matrix(n):
    _R = np.random.uniform(0,1,n*(n-1)/2)
    P = np.zeros((n,n))
    P[np.triu_indices(n, 1)] = _R
    P[np.tril_indices(n, -1)] = P.T[np.tril_indices(n, -1)]
    return P

print(random_symmetric_matrix(N))