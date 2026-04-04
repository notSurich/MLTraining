import pandas as pd
import numpy as np

class MyLogReg:

    def __init__(self,
        n_iter: int = 10,
        learning_rate: float = 0.1,
        weights: np.ndarray = None
        ):

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights


    def __str__(self):
        params = [f'{k}={v}' for k, v in self.__dict__.items()]
        return f"{__class__.__name__} class: " + ", ".join(params)
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        features = X.copy()
        features.insert(loc=0, column='x0', value=1)
        self.weights = np.ones(features.shape[1])
        
        for iter in range(self.n_iter):
            y_pred = 1 / (1 + np.exp(-(features @ self.weights)))
            logloss = self.logloss(y, y_pred)
            gradient = self.gradient(y, y_pred, features)
            self.weights -= self.learning_rate * gradient

            if verbose:
                if iter == 1 or iter % verbose == 0:
                    log = f'{iter} | loss: {logloss}'
                    print(log)


    def logloss(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        loss = (y_true * np.log(y_pred + np.e) + (1 - y_true) * np.log(1 - y_pred + np.e)).mean() * -1
        return loss
    
    def gradient(self, y_true: pd.Series, y_pred: pd.Series, features: pd.DataFrame) -> float:
        grad = 1 / len(y_true) * (y_pred - y_true) @ features
        return grad
    
    def get_coef(self):
        return np.array(self.weights[1:])