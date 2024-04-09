import numpy as np
from abc import abstractmethod
from sklearn.metrics import classification_report, confusion_matrix


class Problem:
    def __init__(self, X, y):
        self.X = np.c_[np.ones(len(X)), np.array(X)]
        self.y = np.array(y).flatten()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.is_trained = False
        self.beta = np.zeros(self.p)
        self.y_pred = None
        self.costs = list()

    @abstractmethod    
    def compute_gradient(self, beta=None):
        pass 

    @abstractmethod
    def compute_cost(self, beta=None):
        pass

    @abstractmethod
    def predict(self, X=None, beta=None):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def evaluate(self, X=None, y_true=None):
        pass


class LeastSqLinearReg(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_gradient(self, beta=None):
        if beta is None:
            beta = self.beta

        self.y_pred = self.predict(beta=beta)
        error = self.y_pred - self.y
        return (self.X.T @ error) / self.n
        
    def compute_cost(self, beta=None):
        if beta is None:
            beta = self.beta

        self.y_pred = self.predict(beta=beta)
        return np.sum(np.square(self.y_pred - self.y)) / (2 * self.n)

    def predict(self, X=None, beta=None):
        use_training_data = X is None
        if use_training_data:
            X = self.X
        else:
            X = np.array(X)
        if len(X.shape) != 2:
            X = X.reshape(-1,1)
        if X.shape[1] == self.p - 1:
            X = np.c_[np.ones(len(X)), np.array(X)]
        elif X.shape[1] != self.p:
            raise ValueError(f'Provided data has {X.shape[1]} column(s); should be {self.p}.')        
        if beta is None:
            beta = self.beta
        
        return X @ beta
    
    def R2(self, y_true, y_pred):    
        return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))

    def Adj_R2(self, r2, obs_no, var_no):
        return 1 - ((1-r2)*(obs_no-1)/(obs_no-var_no-1))

    def evaluate(self, X=None, y_true=None):
        if not ((X is None and y_true is None) or (X is not None and y_true is not None)):
            raise ValueError('Invalid parameters: either provide both values or leave both blank')
                
        y_pred = self.predict(X=X)
        r2 = self.R2(y_true=y_true,y_pred=y_pred)
        print(f'R2 = {r2}')
        print(f'Adj. R2 = {self.Adj_R2(r2,self.n,self.p)}')
    
    def get_name(self):
        return 'LS Linear Regression'


class L2LogisticReg(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = kwargs.get("lambda_", 1.0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_gradient(self, beta=None):
        if beta is None:
            beta = self.beta

        errors = self.sigmoid(self.X @ beta) - self.y
        return ((self.X.T @ errors) + (self.lambda_ * beta)) / self.n

    def compute_cost(self, beta=None):        
        if beta is None:
            beta = self.beta
        self.y_pred = self.predict(beta=beta)

        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        reg_val = (self.lambda_ * np.sum(np.square(beta))) / 2        
        cost = -np.sum((self.y * np.log(y_pred)) + (1 - self.y) * np.log(1 - y_pred))
        return (cost + reg_val) / self.n

    def predict(self, X=None, beta=None):
        use_training_data = X is None
        if use_training_data:
            X = self.X
        else:
            X = np.array(X)
        if len(X.shape) != 2:
            X = X.reshape(-1,1)
        if X.shape[1] == self.p - 1:
            X = np.c_[np.ones(len(X)), np.array(X)]
        elif X.shape[1] != self.p:
            raise ValueError(f'Provided data has {X.shape[1]} column(s); should be {self.p}.')           
        if beta is None:   
            beta = self.beta

        return self.sigmoid(X @ beta)
    
    def evaluate(self, X=None, y_true=None):
        if not ((X is None and y_true is None) or (X is not None and y_true is not None)):
            raise ValueError('Invalid parameters: either provide both values or leave both blank')
        
        self.y_pred = (self.predict(X=X) >= 0.5).astype(int)
        confusion_matrix(y_pred=self.y_pred, y_true=y_true)
        print(classification_report(y_pred=self.y_pred,y_true=y_true))

    def get_name(self):
        return 'L2 Logistic Regression'
