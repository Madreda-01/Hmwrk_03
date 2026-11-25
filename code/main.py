import numpy as np

def prediction_matrices(T):
    At = np.zeros((2*T, T))
    Ct = np.zeros((2*T, 2))
    for k in range(1, T+1):
        Ct[2*(k-1):2*k, :] = [[1, k], [0, 1]]   # N^k
        for j in range(k):                       # j = 0..k-1
            At[2*(k-1):2*k, j] = [k-1-j, 1]      # N^(k-1-j) b
    I2 = np.eye(2)
    return np.kron(At, I2), np.kron(Ct, I2)