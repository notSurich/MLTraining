import numpy as np
import pandas as pd
from typing import Union, Callable
import random


class MyLineReg():

    @staticmethod
    def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
        return (y_true - y_pred).abs().mean()


    @staticmethod
    def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
        return ((y_true - y_pred)**2).mean()


    @staticmethod
    def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
        return np.sqrt(((y_true - y_pred)**2).mean())


    @staticmethod
    def r2(y_true: pd.Series, y_pred: pd.Series) -> float:
        return 1 - ((y_true - y_pred)**2).mean() / ((y_true - y_true.mean())**2).mean()


    @staticmethod
    def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        return 100 * ((y_true - y_pred) / y_true).abs().mean()


    @staticmethod
    def get_metric_score(y_true: pd.Series, y_pred: pd.Series, metric: str):
        if metric == 'mae':
            return MyLineReg.mae(y_true, y_pred)
        elif metric == 'mse':
            return MyLineReg.mse(y_true, y_pred)
        elif metric == 'rmse':
            return MyLineReg.rmse(y_true, y_pred)
        elif metric == 'r2':
            return MyLineReg.r2(y_true, y_pred)
        elif metric == 'mape':
            return MyLineReg.mape(y_true, y_pred)


    def loss(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        loss = ((y_true - y_pred)**2).mean()
        if self.reg == 'l1':
            loss += self.l1_coef * np.abs(self.weights).sum()
        elif self.reg == 'l2':
            loss += self.l2_coef * np.square(self.weights).sum()
        elif self.reg == 'elasticnet':
            loss += self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * np.square(self.weights).sum()
        return loss


    def gradient(self, X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, batch_idx: list) -> float:
        y_true = y_true.iloc[batch_idx]
        y_pred = y_pred[batch_idx]
        X = X.iloc[batch_idx]

        gradient = 2 / len(y_true) * np.dot((y_pred - y_true), X)

        if self.reg == 'l1':
            gradient += self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            gradient += self.l2_coef * 2 * self.weights
        elif self.reg == 'elasticnet':
            gradient += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights 

        return gradient

    def __init__(self,
                n_iter: int = 100,
                learning_rate: Union[float, Callable] = 0.01,
                weights: np.ndarray = None,
                metric: str = None,
                reg: str = None,
                l1_coef: float = 0,
                l2_coef: float = 0,
                sgd_sample: Union[int, float] = None,
                random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state


    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        random.seed(self.random_state)
        features = X.copy()
        features.insert(loc=0, column='x0', value=1)
        self.weights = np.ones(features.shape[1])

        for i in range(1, self.n_iter + 1):
            if isinstance(self.sgd_sample, int):
                batch_idx = random.sample(range(features.shape[0]), self.sgd_sample)
            elif isinstance(self.sgd_sample, float):
                batch_idx = random.sample(range(features.shape[0]), int(features.shape[0] * self.sgd_sample))
            else:
                batch_idx = list(range(features.shape[0]))
            
            y_pred = np.dot(features, self.weights)
            loss = self.loss(y, y_pred)
            gradient = self.gradient(features, y, y_pred, batch_idx)

            if callable(self.learning_rate):
                self.weights -= self.learning_rate(i) * gradient
            else:
                self.weights -= self.learning_rate * gradient
            self.score = MyLineReg.get_metric_score(y, self.predict(X), self.metric)

            if verbose:
                if (i == 1) or (i % verbose == 0):
                    log = f'{i} | loss: {loss}'
                    if self.metric:
                        log += f' | metric: {MyLineReg.get_metric_score(y, y_pred, self.metric)}'
                    print(log)


    def predict(self, X: pd.DataFrame):
        features = X.copy()
        features.insert(loc=0, column='x0', value=1)
        y_pred = np.dot(features, self.weights)
        return y_pred


    def get_coef(self):
        return self.weights[1:]


    def __repr__(self):
        params = [f'{k}={v}' for k,v in self.__dict__.items()]
        return 'MyLineReg class: ' + ', '.join(params)


    def get_best_score(self):
        return self.score
