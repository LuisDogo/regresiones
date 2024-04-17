"""
Adaline (Adaptive Linear Neural) 

@author: carolinacorral
"""
import numpy as np


def adaline(x: np.array,s:np.array,w:np.array,a: float,l:int):
    """
    Adaline 

    Params
    ------
    x: numpy.array 
    Arreglo de variables de entrada 
    s: numpy.array
    Arreglo de variables de salida esperadas
    w: numpy.array
    Pesos iniciales (valores entre 0 y 1 no inclusivo)
    a: float
    Tasa de aprendizaje
    l: int
    Número de épocas (iteraciones) a realizar

    Returns
    ------
    w: numpy.array
    Arreglo de los pesos finales

    """
    #L iteraciones
    for j in range(0,l):
        for i in range(0,len(x)):
            e=[]
            d= s[i]
            y=sum(x[i]*w)
            e.append(d-y)
            if sum(e) == 0:
                return w
            else:
                for j in range(0,len(w)):
                    w[j]+= a*(d-y)*x[i][j]
                
    return w

#Salidas esperadas
s = np.array([1,2,3,4,5,6,7])
#Matriz de entradas
entradas = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

#Pesos iniciales (aleatorios, cercanos a 0)
w1 = 0.84
w2 = 0.39
w3 = 0.78
w= np.array([w1,w2,w3])

pesos=adaline(entradas,s,w,0.3,10)
print(f'W1: {round(pesos[0],2)}, W2: {round(pesos[1],2)}, W3: {round(pesos[2],2)}')