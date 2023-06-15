import numpy as np
from scipy import optimize

def method_choice():
    print("The minimization methods available are: Gradient Descent(GD), Newton(NT), BFGS(BF),and SR1(SR).")
    method = input("choose and type desired minimization method, GD,NT,BF or SR:")
    return method

def get_gradient(f,x):
    return optimize.approx_fprime(x, f, epsilon=1e-12)

def wolfe_condition(f, grad, x, alpha=1.0, c1=0.5, beta=0.8):
    while f(x + alpha*grad) > f(x)+c1*alpha*np.dot((-grad).T, grad):
        alpha *= beta
    return alpha

def gradient_descent(f,x,obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    for i in range(max_iter):
        grad = -get_gradient(f,x)
        
#        if (np.isclose(np.linalg.norm(grad),np.zeros_like(x)).all()):
#            flag = True
#            return x,f(x),flag, path
        
        step = wolfe_condition(f,grad,x)
        new_x = x + step*grad
        
        if (np.linalg.norm(step*grad) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path
        
        path.append(new_x)
        print('For Gradient descent, the final iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
        x = new_x
    return x,f(x),flag, path


def get_hessian(f, x):
    n = len(x)
    hessian = np.zeros((n, n))
    eps = 1e-6

    for i in range(n):
        for j in range(n):
            f_xx = (f(x + eps*np.eye(n)[i] + eps*np.eye(n)[j]) - f(x + eps*np.eye(n)[i]) -
                    f(x + eps*np.eye(n)[j]) + f(x)) / (eps**2)
            hessian[i, j] = f_xx

    return hessian

def newton(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    for i in range(max_iter):
        grad = -get_gradient(f,x)
        hessian = get_hessian(f, x)
        if np.linalg.det(hessian) == 0:
            return x,f(x),flag, path
        
        direction = np.linalg.solve(hessian, grad)
        step = wolfe_condition(f,direction,x)
        new_x = x + step*direction

        if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        #print('the iteration was', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
        x = new_x
    return x,f(x),flag, path

def BFGS(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    #initial B
    B=np.eye(len(x))
    for i in range(max_iter):
        grad = get_gradient(f,x)

        #B_kd=-gk
        direction = -np.linalg.solve(B, grad)
        step = wolfe_condition(f,direction,x)
        p = step*direction
        new_x = x + p

        if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        #print('the iteration was', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

        #update B
        q = get_gradient(f,new_x)-get_gradient(f,x)
        dB = (np.outer(q, q) / np.dot(q, p)) - (np.outer(B @ p, B @ p) / np.dot(p, B @ p))

        #dB = np.dot(q,q.T)/np.dot(q.T,p)-np.dot(np.dot(np.dot(B,p),p.T),B)/np.dot(np.dot(p.T,B),p)
        B = B + dB
        x = new_x

    return x,f(x),flag, path


def SR1(f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    path=[]
    #record start point of path
    path.append(x)
    flag = False
    #initial B
    B=np.eye(len(x))
    for i in range(max_iter):
        grad = get_gradient(f,x)

        direction = -np.dot(B,grad)
        step = wolfe_condition(f,direction,x)
        p = step*direction
        new_x = x + p
        if (np.linalg.norm(p) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
            flag = True
            return x,f(x),flag, path

        path.append(new_x)
        #print('the iteration was', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

        #update B
        q = get_gradient(f,new_x)-get_gradient(f,x)

        dB = np.outer(p - np.dot(B, q), p - np.dot(B, q)) / np.dot(p - np.dot(B, q), q)
        #dB = np.dot((p-np.dot(B,q)),(p-np.dot(B,q)).T)/np.dot((p-np.dot(B,q)).T,q)
        B = B + dB
        x = new_x

    return x,f(x),flag, path




#     pass
# class LineSearch:
#     eps = np.finfo(float).eps
#     max_cond = 1/ eps

#     def __init__(self, step_len=0.1, obj_tol=1e-12, param_tol=1e-8, max_iter=100, calc_hessian=False, wolfe_c1=0.01, wolfe_backtracking=0.5, verbose=True, print_every=4):
#         self.step_len = step_len
#         self.obj_tol = obj_tol
#         self.param_tol = param_tol
#         self.max_iter = max_iter
#         self.calc_hessian = calc_hessian
#         self.verbose = verbose
#         self.print_every = print_every
#         self.wolfe_c1 = wolfe_c1
#         self.wolfe_backtracking = wolfe_backtracking
#         self.run_mode_label = ""
#         self.location_values = None
#         self.objective_values = []
#         self.run_param = {}



#     def wolfe_condition(self,f, grad, x, alpha=1.0, c1=0.5, beta=0.8):
#         while f(x + alpha*grad) > f(x)+c1*alpha*np.dot((-grad).T, grad):
#             alpha *= beta
#         return self.alpha
    
#     def gradient_descent(self,f,x,obj_tol=1e-12, param_tol=1e-8, max_iter=100):
#         path=[]
#         #record start point of path
#         path.append(x)
#         #I wanna change this flag to succ/fail
#         flag = False
#         for i in range(max_iter):
#             grad = -optimize.approx_fprime(x, f, epsilon=1e-12)
#             step = self.wolfe_condition(f,grad,x)
#             new_x = x + step*grad
            
#             if (np.linalg.norm(step*grad) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
#                 flag = True
#                 return self.x,self.f(x),self.flag,self.path
            
#             self.path.append(new_x)
#             #print('the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
#             x = new_x
#         return self.x,self.f(x),self.flag,self.path
    
#     def get_hessian(self,f, x):
#         n = len(x)
#         hessian = np.zeros((n, n))
#         hess_eps = 1e-6

#         for i in range(n):
#             for j in range(n):
#                 fll = (f(x + hess_eps*np.eye(n)[i] + hess_eps*np.eye(n)[j]) - f(x + hess_eps*np.eye(n)[i]) -
#                         f(x + hess_eps*np.eye(n)[j]) + f(x)) / (hess_eps**2)
#                 hessian[i, j] = fll

#         return self.hessian
    
#     def newton(self,f,x, obj_tol, param_tol, max_iter):
#         path=[]
#         #record start point of path
#         path.append(x)
#         flag = False
#         for i in range(max_iter):
#             grad = -self.get_gradient(f,x)
#             hessian = self.get_hessian(f, x)
#             if np.linalg.det(hessian) == 0:
#                 return x,f(x),flag, path
            
#             direction = np.linalg.solve(hessian, grad)
#             step = self.wolfe_condition(f,direction,x)
#             new_x = x + step*direction

#             if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
#                 flag = True
#                 return self.x,self.f(x),self.flag,self.path

#             path.append(new_x)
#             #print('the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))
#             x = new_x
#         return self.x,self.f(x),self.flag,self.path
    
#     def BFGS(self,f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
#         path=[]
#         #record start point of path
#         path.append(x)
#         flag = False
#         #initial B
#         B=np.eye(len(x))
#         for i in range(max_iter):
#             grad = self.get_gradient(f,x)

#             #B_kd=-gk
#             direction = -np.linalg.solve(B, grad)
#             step = self.wolfe_condition(f,direction,x)
#             p = step*direction
#             new_x = x + p

#             if (np.linalg.norm(step*direction) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
#                 flag = True
#                 return x,f(x),flag, path

#             path.append(new_x)
#             #print('the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

#             #update B
#             q = self.get_gradient(f,new_x)-self.get_gradient(f,x)
#             dB = (np.outer(q, q) / np.dot(q, p)) - (np.outer(B @ p, B @ p) / np.dot(p, B @ p))

#             #dB = np.dot(q,q.T)/np.dot(q.T,p)-np.dot(np.dot(np.dot(B,p),p.T),B)/np.dot(np.dot(p.T,B),p)
#             B = B + dB
#             x = new_x

#         return self.x,self.f(x),self.flag,self.path
    
#     def SR1(self,f,x, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
#         path=[]
#         #record start point of path
#         path.append(x)
#         flag = False
#         #initial B
#         B=np.eye(len(x))
#         for i in range(max_iter):
#             grad = self.get_gradient(f,x)

#             direction = -np.dot(B,grad)
#             step = self.wolfe_condition(f,direction,x)
#             p = step*direction
#             new_x = x + p
#             if (np.linalg.norm(p) < param_tol) or (np.abs(f(new_x)-f(x)) < obj_tol):
#                 flag = True
#                 return x,f(x),flag, path

#             path.append(new_x)
#             #print('the iteration is', i+1, 'the current location is', new_x, 'and the current objective value is', f(new_x))

#             #update B
#             q = self.get_gradient(f,new_x)-self.get_gradient(f,x)

#             dB = np.outer(p - np.dot(B, q), p - np.dot(B, q)) / np.dot(p - np.dot(B, q), q)
#             #dB = np.dot((p-np.dot(B,q)),(p-np.dot(B,q)).T)/np.dot((p-np.dot(B,q)).T,q)
#             B = B + dB
#             x = new_x

#         return self.x,self.f(x),self.flag,self.path





