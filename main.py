import numpy as np
from regresion_normal import normal as regnormal
from gradiente import reg_gradiente as reggrad
from regresion_multiple_cuadrada import regresion_multiple as regmul
#from sklearn import linear_model

def main():
    regnormal()
    reggrad()
    regmul()

if __name__ == "__main__":
    main()
