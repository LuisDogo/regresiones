"""
Regresión Lineal con Gradiente Descendiente

@author: carolinacorral
"""
import pandas as pd
import matplotlib.pyplot as plt

def reg_gradiente(x,y,l: float = 0.0001,epochs: int = 1000):
    """
    Params
    ------
    x: numeric array
    Arreglo numérico de variable independiente
    y: anumeric rray
    Arreglo numérico de variable dependiente
    l: float
    Tasa de aprendizaje
    epochs:int
    Épocas (interaciones) a realizar

    Returns
    ------
    m: float
    Pendiente obtenida 
    c: float
    Ordenada al origen obtenida


    """
    m=0
    c=0
    n = len(x)
    for i in range(epochs):
        y_pred = m*x + c
        d_m = (-2/n) * sum(x* (y - y_pred))  # Derivada con respecto a m
        d_c = (-2/n) * sum(y - y_pred)  # Derivada con respecto a c
        m = m - l*d_m #Nueva m
        c = c - l*d_c #Nueva c
    return m,c

#Cargar datos
data = pd.read_csv('data/Salary_Data.csv')
x = data['YearsExperience']
y = data['Salary']

#Obtener m y c con L = 0.001 y 1000 epocas
m, c = reg_gradiente(x,y,0.001,1000)

Y_pred = m*x + c

plt.scatter(x, y) 
plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.title("Regresión con Gradiente Descendiente")
plt.show()
