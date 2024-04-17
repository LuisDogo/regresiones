import numpy as np

def regresion_multiple(X, b):
    # y  = mx +b == y = b0 + b1x + b2x
    sigma = np.empty([len(X),1])
    for column in range(len(X)):
        sigma[column] = sum(X[:, column])
    A = np.outer(sigma.T,sigma)
    return np.linalg.solve(A,b)


a = np.array([[1, 3, 5],[1, 3, 5]])
b = np.array([1, 3])
regresion_multiple(a,b)