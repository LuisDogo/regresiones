import numpy as np
from sklearn import linear_model

def normal(X,y):
    # theta = (XtX)(Xty)
    return np.dot(np.linalg.inv(np.dot(X.T,X)),(np.dot(X.T,y)))

def regresion_lineal_normal(X, y):

np.sum()


