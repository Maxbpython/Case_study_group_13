import pandas as pd
import numpy as np
import copy
from scipy.optimize import minimize, LinearConstraint
import warnings
import statsmodels.api as sm
from pystoned.constant import CET_ADDI, CET_MULT, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned.plot import plot2d
from CNLS_alg import CNLS_LASSO
warnings.filterwarnings('ignore')

def sample_uniform_parameters(
    i, # Nr. Paramaters
    k, # Nr. DMUs
    min_value=10,
    max_value=20,
    seed=42
):
    np.random.seed(seed)
    return pd.DataFrame(
        data=np.sort(np.random.uniform(
            low=min_value,
            high=max_value,
            size=(i, k),
        ))
    )

def sample_correlated_parameters(
    i, # Nr. Paramaters
    k, # Nr. DMUs
    rho,
    min_value=10,
    max_value=20,
    seed=42
):
    x = sample_uniform_parameters(i, k, min_value, max_value, seed=seed)
    for n in range(1,x.shape[1]):
        for j in range(x.shape[1]-1):
            if n != j:
                w = np.random.uniform(10, 20)
                x.loc[:, n] = rho * x.loc[:, j] + w * np.sqrt(1 - rho**2)
    return x

def add_random_variables(
    x,
    i,
    min_value=10,
    max_value=20,
    seed=50
):
    k = len(x.T)
    np.random.seed(seed)
    return pd.concat([x, sample_uniform_parameters(i,k, min_value=min_value, max_value = max_value, seed=seed)], axis=0).reset_index(drop=True)

def output_from_parameters(
    x,
    cons = 0
):
    # y= {}
    # for k in range(x.shape[1]):
    #     y[k] = (x[k]**(1/(len(x)+1))).prod()
    
    y_log = cons + pd.DataFrame((1/(len(x)+1))*np.log(x).sum()).T
        
    return pd.DataFrame(y_log, index=[0])

def output_from_parameters_with_noise(
    x,
    cons = 0,
    var=0.7
):
    y = output_from_parameters(x, cons=cons)
    np.random.seed(42)
    # y= y* np.exp(pd.DataFrame(np.abs(np.random.normal(0, var, size=(y.shape[0],y.shape[1]))).T)
    error = np.abs(np.random.normal(0, var, size=(y.shape[0],y.shape[1])))
    y = pd.DataFrame(y.values - error)
    return y

def theta_objective(
    params
):
    return -params[0]

def obtain_theta(
    x,
    y
):
    """
    The optimization function for theta
    Note that this function will only work if all X's (input variables) are positive
    """
    x_constraint_matrix = copy.deepcopy(x)
    x_constraint_matrix.insert(0, 'theta_cons', 0)

    K = x.shape[1]
    N = x.shape[0]
    Theta_0 = {}
    for O in range(K):
        y_0 = y.loc[:, O][0]
        y_constraint_matrix = copy.deepcopy(y)
        y_constraint_matrix.insert(0, 'theta_cons', -y_0)

        linear_constraint_lambda_x_y = LinearConstraint(
            ############ CONSTRAINT MATRIX ############ 
            # Constraint matrix for Lambda_k*X_k
            x_constraint_matrix.values.tolist()+
            # Constraint matrix for Theta*Y_0 -Lambda_k*Y_k
            (-y_constraint_matrix).values.tolist()+
            # Constraint matrix for sum(Lambda_k)
            [[0] + [1 for i in range(K)]],
            ############### LOWER BOUND ###############
            # Lower bound for X: 0 <= Lambda_k*X_k 
            [0 for i in range(N)]+
            # Lower bound for Y: -infinity <= Theta*Y_0 -Lambda_k*Y_k
            [-np.inf]+
            # Lower bound for Lambda_k: 1 == Sum(Lambda_k)
            [1],
            ############### UPPER BOUND ###############
            # Upper bound for: Lambda_k*X_k <= X_0
            [x_0 for x_0 in x.loc[:, O]]+
            # Upper bound for: Theta*Y_0 -Lambda_k*Y_k <= 0
            [0]+
            # Upper bound for Lambda_k: Sum(Lambda_k) == 1
            [1])
        
        # Initialize the [theta] and [lambda] for the optimization of theta
        x0 = [0]+[1/K for i in range(K)]
        Theta_0[O] = minimize(theta_objective, x0, method='trust-constr',constraints=[linear_constraint_lambda_x_y],options={'disp': False})['x'][0]
    return pd.DataFrame(Theta_0, index=[0])


def create_var_matrix(
    x,
    y
):
    x_mat = x.T.rename(columns=lambda x: 'X_'+str(x))
    x_mat.insert(0, 'intercept', 1)

    var_mat = pd.concat([y.T, x_mat], axis=1).rename(columns={0: 'y'})
    var_mat.insert(1, 'y_hat', '')
    var_mat.insert(2, 'residual', '')
    var_mat.insert(3, 'residual_squared', '')
    return x_mat, var_mat

def calculate_SSR_plus_LASSO(
    beta
):
    global var_mat
    global x_mat
    global eta
    
    var_mat['y_hat'] = x_mat@beta
    var_mat['residual'] = var_mat['y'] - var_mat['y_hat']
    var_mat['residual_squared'] = var_mat['residual']**2

    # Calculate the statistics
    SSR = var_mat['residual_squared'].sum()
    LASSO = eta*np.abs(beta[1:]).sum()
    return SSR+ LASSO

def perform_CNLS_LASSO(
    x,
    y,
    eta,
):
    """
    Perform the CNLS with LASSO
    """
    x_T = x.T.values
    y_log = y.T.values
    # y_log = np.log(output_from_parameters_with_noise(x).T.values)

    model = CNLS_LASSO(y_log, x_T, z=None, eta=eta, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize('maxklaasbakker@gmail.com')
    return model

def objective_beta_unique(
    params
):
    global X_O
    return params[0] + sum([x * y for x, y in zip(params[1:], X_O)])

def beta_constraints(
    beta,
    alpha,
    x
    ):
    return alpha +(x.T*beta).sum(axis=1)

def obtain_beta_unique(
    x,
    alpha,
    beta
):
    """
    This function will give the lower bound for beta so that we get a unique value for beta
    This is done by minimizing the objective function with the given constraints
    Reasoning behind this is given in the report
    """
    constraint_beta = beta_constraints(beta, alpha,x)

    K = x.shape[1]
    I = x.shape[0]
    global X_O

    Beta_0 = {}
    for O in range(K):
        X_O = x[O] 
        constraint_1 = constraint_beta[O]
        linear_constraint_alpha_beta = LinearConstraint(
            ############ CONSTRAINT MATRIX ############
            # Constraint matrix for alpha+sum(beta*X)
            [[-1]+(-x[k].values).tolist() for k in range(K)]+ 
            # Constraint matrix for Beta
            np.array([np.zeros(I).tolist()]+ np.identity(I).tolist()).T.tolist(),
            ############### LOWER BOUND ###############
            [-np.inf]*K+np.zeros(I).tolist(),
            ############### UPPER BOUND ###############
            (-constraint_beta).tolist()+
            [beta[O].sum() for i in range(I)])
            
        
        # Initialize the [alpha] and [beta] for the optimization of theta
        x0 = [alpha[O]]+beta[O].tolist()
        Beta_0[O] = minimize(objective_beta_unique, x0, method='trust-constr',constraints=[linear_constraint_alpha_beta],options={'disp': False})['x']
    return pd.DataFrame(Beta_0)