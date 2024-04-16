import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    clipped_x = np.clip(x, -500, 500)  # Limita los valores de x para evitar desbordamiento
    return 1 / (1 + np.exp(-clipped_x))

def cost_function(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Limita las probabilidades predichas para evitar divisiones por cero
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent(X: np.ndarray, y: np.ndarray,num_iters: int, learning_rate: float, ):
    num_samples, num_features = X.shape

    weights = np.zeros(num_features)
    bias = 0

    for i in range(num_iters):
        y_pred = sigmoid_function( np.dot(X,weights,) + bias)

        dw = ( 1 / num_samples ) * np.dot(X.T, (y_pred - y))
        db = ( 1 / num_samples ) * np.sum(y_pred - y)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        cost = cost_function(y,y_pred)

    return weights, bias

def predict(X: np.ndarray, weights: np.ndarray, bias: float):
    y_pred = sigmoid_function(np.dot(X,weights) + bias)
    y_pred_bin = [1 if x >= 0.5 else 0 for x in y_pred]
    return y_pred,y_pred_bin

def encoding_gender(gender: str, diccionary_gender: dict) -> int:
    return diccionary_gender.get(gender)

def run():
    df = pd.read_csv("/home/gsu/ESCOM/regresiones/data/regresionLogistica.csv")
    df.drop('User ID',axis=1, inplace=True)

    diccionary_gender = {"Male": 0, "Female": 1}

    df["Gender"] = df["Gender"].apply(lambda x: encoding_gender(x, diccionary_gender))

    y = df["Purchased"]
    
    X = df.drop("Purchased", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    learning_rate = 0.001

    n_iters = 1000

    weights, bias = gradient_descent(X_train,y_train, n_iters,learning_rate)
    
    y_pred, y_pred_bin = predict(X_test, weights,bias)

    accuracy_custom = accuracy_score(y_test, y_pred_bin)

    print(f"Accuracy: {accuracy_custom}")
   

if '__main__' == __name__:
    run()