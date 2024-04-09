import numpy as np
from numpy.linalg import norm, inv
from abc import abstractmethod
from tqdm import tqdm
from enum import Enum

from problems import * 


class Algorithm:
    def __init__(self, problem:Problem):
        self.problem = problem

    @abstractmethod
    def fit(self, learning_rate, max_iterations, tol):
        pass
        
    @abstractmethod
    def get_name(self):
        pass
   

class GradientDescent(Algorithm):
    def __init__(self, problem:Problem):
        super().__init__(problem)

    def fit(self, learning_rate = 0.01, max_iterations = 1000, tol=1e-18):   
        print(f'{self.get_name()}  - {self.problem.get_name()}')

        beta = np.ones(self.problem.p) * 1e-6
        best_beta = beta
        best_cost = -1
        self.problem.costs.append(self.problem.compute_cost(beta=beta))

        if learning_rate <= 0:
            learning_rate = 0.01

        progress_bar = tqdm(total=max_iterations, desc="Progress")            
        for _ in range(max_iterations):
            gradient = self.problem.compute_gradient(beta=beta)

            if np.isinf(gradient).any() or np.isnan(gradient).any() or norm(gradient) <= tol:
                break            

            beta -= learning_rate * gradient
            
            new_cost = self.problem.compute_cost(beta=beta)
            self.problem.costs.append(new_cost)
            
            if new_cost < best_cost or best_cost < 0:
                best_beta = beta
                best_cost = new_cost
            
            progress_bar.update(1)

        progress_bar.close()

        self.problem.beta = best_beta
        self.problem.is_trained = True

    def get_name(self):
        return 'Gradient Descent'


class QuasiNewton(Algorithm):
    def __init__(self, problem:Problem):
            super().__init__(problem)
            self.qn_violation = list()

    @abstractmethod
    def update_hessian(self, hessian_approx, s, y):
        pass

    def wolfe_search(self, direction, beta, gamma=1e-6, delta=0.5, sigma=0.9, alpha0=5.0, max_iterations = 50, use_strong_wolfe=True, early_stopping=True, tol=1e-4, min_step=1e-8):
        alpha = alpha0
        
        cost = self.problem.compute_cost(beta=beta)
        gradient = self.problem.compute_gradient(beta=beta)
        grad_d = (gradient.T @ direction)

        for _ in range(max_iterations):
            beta_new = beta + alpha * direction

            cost_new = self.problem.compute_cost(beta=beta_new)
            gradient_new = self.problem.compute_gradient(beta=beta_new)

            suff_decr = cost + gamma * alpha * grad_d
            armijo_cond = cost_new <= suff_decr or (early_stopping and np.abs(cost_new - suff_decr) <= tol)

            gradn_d = (gradient_new.T @ direction)

            if use_strong_wolfe:
                gradn_d = np.abs(gradn_d)
                w_cond = sigma * np.abs(grad_d)
            else:
                gradn_d = -gradn_d
                w_cond = -sigma * grad_d
                
            curvature_cond = gradn_d <= w_cond or (early_stopping and np.abs(gradn_d - w_cond) <= tol)

            if armijo_cond and curvature_cond or (early_stopping and alpha < min_step):
                break

            alpha *= delta
        
        return alpha
    
    def check_qn_condition(self, H, s, y):
        violation = norm(H @ y - s)
        self.qn_violation.append(violation)
        return violation


class BFGS(QuasiNewton):
    def __init__(self, problem:Problem):
        super().__init__(problem)
    
    def update_hessian(self, H, s, y):
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)    
        return H + ((s.T @ y + y.T @ H @ y) * (s @ s.T))/np.square(s.T @ y) - (H @ y @ s.T + s @ y.T @ H) / (s.T @ y)

    def fit(self, learning_rate=0, max_iterations = 1000, tol=1e-18):        
        print(f'{self.get_name()} - {self.problem.get_name()}')
        
        beta = np.ones(self.problem.p) * 1e-6
        best_beta = beta
        best_cost = -1
        self.problem.costs.append(self.problem.compute_cost(beta=beta))
            
        gradient = self.problem.compute_gradient(beta=beta)
        use_wolfe = learning_rate <= 0        
        
        hessian=np.identity(self.problem.p)
        y = np.zeros(self.problem.p)
        s = np.zeros(self.problem.p)

        progress_bar = tqdm(total=max_iterations, desc="Progress")
        for _ in range(max_iterations):                        

            if np.isinf(gradient).any() or np.isnan(gradient).any() or norm(gradient) <= tol:
                break
            
            direction = - hessian @ gradient
            if use_wolfe:
                learning_rate = self.wolfe_search(direction=direction, beta=beta)
            
            beta_new = beta + learning_rate * direction
            
            new_grad = self.problem.compute_gradient(beta=beta_new)
            y = new_grad - gradient
            s = beta_new - beta
            
            if np.isinf(beta_new).any() or np.isnan(beta_new).any() or norm(s) <= tol:
                break        

            beta = beta_new
            gradient = new_grad
            
            hessian = self.update_hessian(hessian, s, y)

            self.check_qn_condition(H=hessian,s=s,y=y)

            new_cost = self.problem.compute_cost(beta=beta)
            self.problem.costs.append(new_cost)
            
            if new_cost < best_cost or best_cost < 0:
                best_beta = beta
                best_cost = new_cost

            progress_bar.update(1)

        progress_bar.close()
        self.problem.beta = best_beta 
        self.problem.is_trained = True
    
    def get_name(self):
        return 'BFGS'


class AMSQN_Mode(Enum):
    EXACT = 0
    SYMMETRY = 1
    POS_DEF = 2
    BOTH = 3


class AMSQN(QuasiNewton):
    def __init__(self, problem:Problem, mode:AMSQN_Mode=AMSQN_Mode.BOTH):
        super().__init__(problem)
        self.mode = mode
    
    def find_mu(self, delta):
        eigvals, _ = np.linalg.eig(delta)
        return max(0, np.real(-min(eigvals)))

    def update_hessian(self, B, s, y):
        d1_w_d2 = - ((y @ inv(y.T @ s) @ y.T) - (B @ s @ inv(s.T @ B @ s) @ s.T @ B))
        n, _ = B.shape
                
        match self.mode:
            case AMSQN_Mode.EXACT:
                B = B - d1_w_d2

            case AMSQN_Mode.SYMMETRY:
                B = B - (d1_w_d2 + d1_w_d2.T) / 2.

            case AMSQN_Mode.POS_DEF:
                delta = - d1_w_d2
                mu_I = self.find_mu(delta)  * np.identity(n)
                B = B + delta + mu_I
                
            case AMSQN_Mode.BOTH:
                delta = -(d1_w_d2 + d1_w_d2.T) / 2.
                mu_I = self.find_mu(delta)  * np.identity(n)
                B = B + delta + mu_I

        return B
                
    def fit(self, learning_rate=0.01, max_iterations = 1000, tol=1e-18):
        print(f'{self.get_name()} - {self.problem.get_name()}')
        
        beta = np.ones(self.problem.p) * 1e-6        
        best_beta = beta
        best_cost = -1
        self.problem.costs.append(self.problem.compute_cost(beta=beta))

        gradient = self.problem.compute_gradient(beta=beta)
        
        if self.mode == AMSQN_Mode.BOTH:
            use_wolfe = learning_rate <= 0
        else:
            use_wolfe = False
            if learning_rate <= 0:
                learning_rate = 0.001

        hessian = np.identity(self.problem.p)
        hessian_inv = inv(hessian)
        
        Ys = list()
        Ss = list()
        curr_memory_size = 0
        multi_secant_memory_size=10

        progress_bar = tqdm(total=max_iterations, desc="Progress")                

        for _ in range(1, max_iterations+1):

            if np.isinf(gradient).any() or np.isnan(gradient).any() or norm(gradient) <= tol:
                break
            
            direction = - hessian_inv @ gradient
            if use_wolfe:
                learning_rate = self.wolfe_search(direction=direction, beta=beta)

            beta_new = beta + learning_rate * direction
            
            new_grad = self.problem.compute_gradient(beta=beta_new)
            y = new_grad - gradient
            s = beta_new - beta

            if np.isinf(beta_new).any() or np.isnan(beta_new).any() or norm(s) <= tol or norm(y) <= tol:
                break

            beta = beta_new
            gradient = new_grad
            
            if curr_memory_size == multi_secant_memory_size:
                Ss.pop(0)
                Ys.pop(0)
                curr_memory_size -= 1
            index = min(multi_secant_memory_size - 1, curr_memory_size)
            Ss.insert(index,s)
            Ys.insert(index,y)
            curr_memory_size +=1
            
            hessian = self.update_hessian(B=hessian, s=np.array(Ss).T, y=np.array(Ys).T)
            hessian_inv = inv(hessian)

            self.check_qn_condition(H=hessian_inv, s=np.array(Ss).T, y=np.array(Ys).T)
            
            new_cost = self.problem.compute_cost(beta=beta)
            self.problem.costs.append(new_cost)

            if new_cost < best_cost or best_cost < 0:
                best_beta = beta
                best_cost = new_cost

            progress_bar.update(1)            

        progress_bar.close()
        self.problem.beta = best_beta
        self.problem.is_trained = True
    
    def get_name(self):
        return f'AMSQN - {self.mode.name}'