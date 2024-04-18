import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from regresion_normal import normal as regnormal
from regresion_multiple_cuadrada import regresion_multiple as regmul


def main():
    df = pd.read_csv('data/Salary_Data.csv')
    X = df.to_numpy()[:,0].reshape(-1, 1)
    y = df.to_numpy()[:,1].reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    print(reg.coef_)
    print(regnormal(X,y))

if __name__ == "__main__":
    main()
