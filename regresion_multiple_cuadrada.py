import numpy as np
from sklearn.linear_model import LinearRegression

def regresion_multiple(X, b):
    m = X.shape[1]
    sigma0 = np.empty([m,1])
    for column in range(m):
        sigma0[column] = X[:, column].sum()
    sigma0 = np.insert(sigma0, 0, 1)
    print(sigma0) 
    A = np.outer(sigma0,sigma0)
    A[0,0] = m
    #A = np.outer(sigma0,sigma0)    
    print(sigma0) 
    print(A)
    print("res")
    return np.dot(A,b)


X = np.array([[1, 2, 1], [3, 4, 3], [5, 6, 3], [7, 8, 3]])
y = np.array([3, 7, 11, 15])
print(regresion_multiple(X,y))
