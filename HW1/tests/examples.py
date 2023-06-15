from unconstrained_min import *
from scipy import optimize




#q1
def f1(x):
    q=np.array([[1,0],[0,1]])
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)

#q2
def f2(x):
    q=np.array([[1,0],[0,100]])
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)


# def function2(x,calc_hess=False):
#     f=f2(x)
#     hessian = LineSearch().get_hessian(f,x)
#     g = optimize.approx_fprime(x, f2, epsilon=1e-12)
#     if calc_hess==True:
#         return f,g,hessian
#     return f,g

#q3

def f3(x):
    a=np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]]).T
    b=np.array([[100,0],[0,1]])
    c=np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]])
    q=np.dot(np.dot(a,b),c)
    x = np.array(x)
    return np.dot(np.dot((x.T),q),x)

# def function3(x,bool_v=False):
#     f=f3(x)
#     hessian = LineSearch().get_hessian(f,x)
#     g= optimize.approx_fprime(x, f3, epsilon=1e-12)
#     if bool_v==True:
#         return f,g,hessian
#     return f,g


#Rosenbrock
#calculated hessian manually beforehand

# def rosenbrock(x, calc_hess=False, calc_gradient=True):
#     # x might be a matrix of vertical vectors of (x1,x2)
#     x1, x2 = x[0], x[1]
#     f = 100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2
#     if f.size == 1:
#         f = f.item()
#     if not calc_gradient:
#         return f
#     g = np.vstack([400 * x1 ** 3 - 400 * x1 * x2 + 2 * x1 - 2, 200 * (x2 - x1 * x1)])
#     assert g.shape == x.shape
#     if calc_hess and x.shape[1] == 1:
#         x1, x2 = x1.item(), x2.item()
#         h = np.array([[1200 * x1 * x1 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]], dtype=float)
#     else:
#         h = None
#     return f, g, h

def fRosenbrock(x):
    x = np.array(x)
    q=np.array([[1200*x[0]**2-400*x[1],-400*x[0]],[-400*x[0],200]])
    return np.dot(np.dot((x.T),q),x)

#a linear function ...
def f5(x):
    a=np.array([4,9])
    x = np.array(x)
    return np.dot((a.T),x)


#boyd's
#calculated hessian manually beforehand also

#def boyd(x, calc_hess=False, calc_gradient=True):
    # x might be a matrix of vertical vectors of (x1,x2)
    x1, x2 = x[0], x[1]
    f1 = np.exp(x1 + 3 * x2 - 0.1)
    f2 = np.exp(x1 - 3 * x2 - 0.1)
    f3 = np.exp(-x1 - 0.1)
    f = f1 + f2 + f3
    if f.size == 1:
        f = f.item()
    if not calc_gradient:
        return f
    g2 = 3 * f1 - 3 * f2
    g = np.vstack([f1 + f2 - f3, g2])
    assert g.shape == x.shape
    if calc_hess and x.shape[1] == 1:
        f1 = f1.item()
        f2 = f2.item()
        g2 = g2.item()
        h = np.array([[f, g2], [g2, 9 * f1 + 9 * f2]])
    else:
        h = None
    return f, g, h

def fboyd(x):
    return np.e**(x[0]+3*x[1]-0.1)+np.e**(x[0]-3*x[1]-0.1)+np.e**(-x[0]-0.1)

