import numpy as np
import pandas as pd

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.metrics = {
            'mae':  lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse':  lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2':   lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):

        X_work = X.copy()
        n_samples, n_features = X_work.shape
        
        ones = np.ones(n_samples)
        if isinstance(X_work, pd.DataFrame):
            X_work.insert(0, 'x_0', ones)
        else:
            X_work = np.column_stack((ones, X_work))
        
        if isinstance(self.sgd_sample, float):
            k_samples = int(n_samples * self.sgd_sample)
        elif self.sgd_sample is None:
            k_samples = n_samples
        else:
            k_samples = self.sgd_sample
            
        random.seed(self.random_state)
        
        self.weights = np.ones(X_work.shape[1])
        
        for i in range(1, self.n_iter + 1):
            sample_rows_idx = random.sample(range(n_samples), k_samples)
            
            if isinstance(X_work, pd.DataFrame):
                X_batch = X_work.iloc[sample_rows_idx]
                y_batch = y.iloc[sample_rows_idx] if isinstance(y, pd.Series) else y[sample_rows_idx]
            else:
                X_batch = X_work[sample_rows_idx]
                y_batch = y[sample_rows_idx]
            
            predictions_batch = X_batch @ self.weights
            base_grad_vector = (2.0 / len(y_batch)) * (X_batch.T @ (predictions_batch - y_batch))
            
            final_grad = self.grad_regularisation(base_grad_vector)
            
            if callable(self.learning_rate):
                lr_val = self.learning_rate(i)
            else:
                lr_val = self.learning_rate
            
            self.weights -= lr_val * final_grad
            
            y_pred_full = X_work @ self.weights
            base_loss = np.mean((y - y_pred_full) ** 2)
            current_loss = self.loss_regularisation(base_loss)
            
            if self.metric:
                self.best_score = self.metrics[self.metric](y, y_pred_full)

            if verbose and i % verbose == 0:
                metric_str = f"| {self.metric}: {self.best_score:.4f}" if self.metric else ''
                print(f"{i} | loss: {current_loss:.4f} {metric_str}")
                

    def predict(self, X):
        X_work = X.copy()
        ones = np.ones(X_work.shape[0])
        
        if isinstance(X_work, pd.DataFrame):
            X_work.insert(0, 'x_0', ones)
        else:
            X_work = np.column_stack((ones, X_work))
            
        return X_work @ self.weights
    
    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.best_score
        
    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def loss_regularisation(self, base_loss):
        if self.reg == 'l1':
            penalty = self.l1_coef * np.sum(np.abs(self.weights[1:]))
            return base_loss + penalty
        elif self.reg == 'l2':
            penalty = self.l2_coef * np.sum(self.weights[1:] ** 2)
            return base_loss + penalty
        elif self.reg == 'elasticnet':
            p1 = self.l1_coef * np.sum(np.abs(self.weights[1:]))
            p2 = self.l2_coef * np.sum(self.weights[1:] ** 2)
            return base_loss + p1 + p2
        else:
            return base_loss
    
    def grad_regularisation(self, base_grad):
        penalty_vector = np.zeros_like(self.weights)
        
        if self.reg == 'l1':
            penalty_vector = self.l1_coef * np.sign(self.weights)
            
        elif self.reg == 'l2':
            penalty_vector = self.l2_coef * 2 * self.weights
           
        elif self.reg == 'elasticnet':
            p1 = self.l1_coef * np.sign(self.weights)
            p2 = self.l2_coef * 2 * self.weights
            penalty_vector = p1 + p2
            
        return base_grad + penalty_vector
