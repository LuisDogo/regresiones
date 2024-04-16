import numpy as np

def normal(X,y):
    # theta = (XtX)(Xty)
    return np.dot(np.linalg.inv(np.dot(X.T,X)),(np.dot(X.T,y)))
