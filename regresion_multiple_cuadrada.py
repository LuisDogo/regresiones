"""
Regresión Lineal multiple mediante mínimos cuadrados

@author: luisdogo
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def regresion_multiple(X, b):
    m = X.shape[1] # número de observaciones
    sigma0 = np.empty([m,1])
    for column in range(m):
        sigma0[column] = X[:, column].sum()
    print(sigma0)
    sigma1 = np.insert(sigma0, 0, 1)
    A = np.outer(sigma1,sigma1)
    A[0,0] = m
    return np.dot(A,b)