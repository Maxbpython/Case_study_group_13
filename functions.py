import pandas as pd
import numpy as np
import copy
from scipy.optimize import minimize, LinearConstraint
import warnings
import statsmodels.api as sm
from pystoned.constant import CET_ADDI, CET_MULT, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned.plot import plot2d
from CNLS_alg import CNLS_LASSO
import time
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')

def simulate_run(run, test2):
    SEED = run
    x = sample_uniform_parameters(i=2,k=25,min_value=10, max_value=20, seed=SEED)
    time.sleep(0.1)
    # y = simulate_output(A,x)
    return x

def run_simulation_CNLS_LASSO(run, CORRELATION, CORRELATION_REDUNDANT_VARIABLES, TRUE_INPUTS, REDUNDANT_INPUTS, NR_DMU, ETA, VAR_mu,corrs_TRUE, corrs_FALSE, EMAIL):
    # This function will be fed to the multiprocessing.Pool.map function
    # It will be run in parallel
    
    result_dict = {'SSR':{}, 'MSE':{}, 'nr_variables_deleted':{}, 'nr_correct_variables_deleted':{}}
    SEED = run
    if CORRELATION:
        x = sample_correlated_parameters(i=TRUE_INPUTS,k=NR_DMU, rho_dict=corrs_TRUE, min_value=10, max_value=20, seed=SEED)
    else:
        x = sample_uniform_parameters(i=TRUE_INPUTS,k=NR_DMU, min_value=10, max_value=20, seed=SEED)

    y_log_true = output_from_parameters(x, cons = 3)
    y_log = output_from_parameters_with_noise(x, cons=3, var=VAR_mu)

    if CORRELATION_REDUNDANT_VARIABLES:
        x = add_random_correlated_variables(x, REDUNDANT_INPUTS, corrs_FALSE, min_value = 10, max_value = 20, seed=SEED+1)
    else:
        x = add_random_variables(x, REDUNDANT_INPUTS, min_value = 10, max_value = 20, seed=SEED+1)


    model_cnls = perform_CNLS_LASSO(x=x, y=y_log, eta=ETA, email=EMAIL)
    beta = model_cnls.get_beta()
    alpha = model_cnls.get_alpha()
    beta = pd.DataFrame(beta).round(2)
    beta.loc['Total',:] = beta.mean(axis=0).round(2)
    result_dict['SSR'] = SSR_model = (model_cnls.get_residual()**2).sum()
    result_dict['MSE'] = MSE_model = (model_cnls.get_residual()**2).mean()
    result_dict['nr_variables_deleted'] = nr_variables_deleted = (beta.loc['Total',:] == 0).sum()
    result_dict['nr_correct_variables_deleted'] = nr_correct_variables_deleted = (beta.loc['Total',[0,1]] == 0).sum()
    return result_dict

def run_simulation_CNLS_LASSO_RANDOM(run, CORRELATION, CORRELATION_REDUNDANT_VARIABLES, TRUE_INPUTS, REDUNDANT_INPUTS, NR_DMU, ETA, VAR_mu,corrs_TRUE, corrs_FALSE, eta_reg_LASSO, EMAIL):
    results_SCNLS_LASSO_run = {'SSR':{}, 'MSE':{}, 'nr_variables_deleted':{}, 'nr_correct_variables_deleted':{}}
    results_random_run = {'SSR':{}, 'MSE':{}, 'nr_variables_deleted':{}, 'nr_correct_variables_deleted':{}}
    results_reg_LASSO_run = {'SSR':{}, 'MSE':{}, 'nr_variables_deleted':{}, 'nr_correct_variables_deleted':{}}
    
    SEED = run
    if CORRELATION:
        x = sample_correlated_parameters(i=TRUE_INPUTS,k=NR_DMU, rho_dict=corrs_TRUE, min_value=10, max_value=20, seed=SEED)
    else:
        x = sample_uniform_parameters(i=TRUE_INPUTS,k=NR_DMU, min_value=10, max_value=20, seed=SEED)

    y_log_true = output_from_parameters(x, cons = 3)
    y_log = output_from_parameters_with_noise(x, cons=3, var=VAR_mu)

    if CORRELATION_REDUNDANT_VARIABLES:
        x = add_random_correlated_variables(x, REDUNDANT_INPUTS, corrs_FALSE, min_value = 10, max_value = 20, seed=SEED+1)
    else:
        x = add_random_variables(x, REDUNDANT_INPUTS, min_value = 10, max_value = 20, seed=SEED+1)
    
    x_random, nr_random_deletions, nr_random_true_variables_deleted = delete_random_variables(x, seed=SEED+2)

    # SCNLS-LASSO
    model_scnls_lasso = perform_CNLS_LASSO(x=x, y=y_log, eta=ETA, email=EMAIL)
    ssr, mse, nr_variables_deleted_lasso, nr_correct_variables_deleted_lasso = retrieve_results(model_scnls_lasso)
    results_SCNLS_LASSO_run['SSR'] = ssr
    results_SCNLS_LASSO_run['MSE'] = mse
    results_SCNLS_LASSO_run['nr_variables_deleted'] = nr_variables_deleted_lasso
    results_SCNLS_LASSO_run['nr_correct_variables_deleted'] = nr_correct_variables_deleted_lasso

    # Random deletion
    model_random = perform_CNLS_LASSO(x=x_random, y=y_log, eta=0, email=EMAIL)
    ssr, mse, _, _ = retrieve_results(model_random)
    results_random_run['SSR'] = ssr
    results_random_run['MSE'] = mse
    results_random_run['nr_variables_deleted'] = nr_random_deletions
    results_random_run['nr_correct_variables_deleted'] = nr_random_true_variables_deleted


    # Regular LASSO
    lasso_model = Lasso(alpha=eta_reg_LASSO)
    lasso_model.fit(np.log(x).T, y_log.T)
    ssr, mse, nr_variables_deleted_reg_lasso, nr_correct_variables_deleted_reg_lasso = retrieve_results_regular_lasso(lasso_model, x, y_log)
    results_reg_LASSO_run['SSR'] = ssr
    results_reg_LASSO_run['MSE'] = mse
    results_reg_LASSO_run['nr_variables_deleted'] = nr_variables_deleted_reg_lasso
    results_reg_LASSO_run['nr_correct_variables_deleted'] = nr_correct_variables_deleted_reg_lasso

    return results_SCNLS_LASSO_run, results_random_run, results_reg_LASSO_run

def sample_uniform_parameters(
    i, # Nr. Paramaters
    k, # Nr. DMUs
    min_value=10,
    max_value=20,
    seed=42
):
    np.random.seed(seed)
    return pd.DataFrame(
        data=np.random.uniform(
            low=min_value,
            high=max_value,
            size=(i, k),
        )
    )

def create_corr_mat(rho_inputs:dict, n):
    # minimum number of correlation inputs (rho_inputs) is 2 and maximum is 5. Since variables have to be predefined.
    rho = np.zeros((n, n))
    np.fill_diagonal(rho, 1)
    for i in rho_inputs.keys():
        for j in rho_inputs[i].keys():
            try:
                rho[i-1, j-1] = rho[j-1, i-1] = rho_inputs[i][j]
            except:
                pass
    return rho

def sample_correlated_parameters(
    i, # Nr. Parameters
    k, # Nr. DMUs
    rho_dict,
    min_value=10,
    max_value=20,
    seed=42
):
    x = sample_uniform_parameters(i, k, min_value, max_value)
    rho_mat = create_corr_mat(rho_dict, i)
    np.random.seed(seed)
    for i in range(1,x.shape[0]):
        for j in range(x.shape[0]-1):
            if i != j:
                if rho_mat[i,j] == 0:
                    continue
                w = np.random.uniform(10, 20)
                x.loc[:, i] = rho_mat[i,j] * x.loc[:, j] + w * np.sqrt(1 - rho_mat[i,j]**2)
                
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

def add_random_correlated_variables(
    x,
    i,
    rho_dict,
    min_value=10,
    max_value=20,
    seed=50
):
    k = len(x.T)
    np.random.seed(seed)
    return pd.concat([x, sample_correlated_parameters(i,k,rho_dict=rho_dict, min_value=min_value, max_value = max_value, seed=seed)], axis=0).reset_index(drop=True)

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

def delete_random_variables(x, seed):
    if x.shape[0]==1:
        return x, 0, 0
    np.random.seed(seed)
    nr_deletions = int(np.random.uniform(0,x.shape[0]-1))
    sample_deletions = np.sort(np.random.choice(x.shape[0], nr_deletions, replace=False))
    nr_true_variables_deleted = ((sample_deletions == 0) | (sample_deletions == 1)).sum()
    x.drop(x.index[sample_deletions],inplace=True)
    return x, nr_deletions, nr_true_variables_deleted

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
    email:str
):
    """
    Perform the CNLS with LASSO
    """
    x_T = x.T.values
    y_log = y.T.values
    # y_log = np.log(output_from_parameters_with_noise(x).T.values)

    model = CNLS_LASSO(y_log, x_T, z=None, eta=eta, cet = CET_ADDI, fun = FUN_PROD, rts = RTS_VRS)
    model.optimize(email)
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

def retrieve_results(model_cnls):
    beta = model_cnls.get_beta()
    beta = pd.DataFrame(beta).round(2)
    beta.loc['Total',:] = beta.mean(axis=0).round(2)
    if beta.shape[1] == 1:
        true_params = [0]
    else:
        true_params = [0,1]

    ssr = (model_cnls.get_residual()**2).sum()
    mse = (model_cnls.get_residual()**2).mean()
    nr_variables_deleted =(beta.loc['Total',:] == 0).sum()
    nr_correct_variables_deleted = (beta.loc['Total',true_params] == 0).sum()
    return ssr, mse, nr_variables_deleted, nr_correct_variables_deleted

def retrieve_results_regular_lasso(model_lasso, x,y_log):
    beta = model_lasso.coef_
    beta = pd.DataFrame(beta).T.round(3)
    if beta.shape[1] == 1:
        true_params = [0]
    else:
        true_params = [0,1]

    ssr = ((y_log.T - pd.DataFrame(model_lasso.predict(np.log(x).T)))**2).sum()[0]
    mse = ((y_log.T - pd.DataFrame(model_lasso.predict(np.log(x).T)))**2).mean()[0]
    nr_variables_deleted = (beta.loc[0,:]==0).sum()
    nr_correct_variables_deleted = (beta.loc[0,true_params]==0).sum()
    return ssr, mse, nr_variables_deleted, nr_correct_variables_deleted

def perform_grid_search_reg_LASSO(alphas=np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]), reps=30):
    #import packages
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV

    #create grid search
    grid_search = GridSearchCV(estimator=Lasso(), param_grid={'alpha':alphas}, cv=5, scoring='neg_mean_squared_error')

    #run grid search for each dataset and store results
    all_results = []

    for run in range(0,reps):
        SEED = run
        if CORRELATION:
            x = sample_correlated_parameters(i=TRUE_INPUTS,k=NR_DMU, rho_dict=corrs, min_value=10, max_value=20, seed=SEED)
        else:
            x = sample_uniform_parameters(i=TRUE_INPUTS,k=NR_DMU, min_value=10, max_value=20, seed=SEED)

        y_log = output_from_parameters_with_noise(x, cons=3, var=0.1)

        if CORRELATION_REDUNDANT_VARIABLES:
            x = add_random_correlated_variables(x, REDUNDANT_INPUTS, corrs, min_value = 10, max_value = 20, seed=SEED+1)
        else:
            x = add_random_variables(x, REDUNDANT_INPUTS, min_value = 10, max_value = 20, seed=SEED+1)
        
        grid_search.fit(np.log(x).T, y_log.T)
        all_results.append(grid_search.best_params_['alpha'])

    #calculate average alpha
    average_alpha = np.mean(all_results)
    return average_alpha