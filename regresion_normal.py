"""
Regresión Lineal mediante ecuación normal

@author: luisdogo
"""
import numpy as np

def normal(X,y):
    # (Xt X)-1 (Xt y)
    # y = b0 + b1x1 + b2x2 + ... + bnxn
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return (np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y))