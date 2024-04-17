import numpy as np

def regresion_multiple(X, b):
    sigma0 = np.empty([len(X),1])
    for column in range(len(X)):
        sigma0[column] = sum(X[:, column])
    sigma1 = np.insert(sigma0, 0, 1)
    #A = np.outer(sigma1,sigma1)
    A = np.outer(sigma0,sigma0)    
    print(A)
    print("res")
    return np.dot(A,b)


a = np.array([[3, 7, 5],[1, 3, 0]])
b = np.array([1, 3])
print(regresion_multiple(a,b))