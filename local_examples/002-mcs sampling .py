import sys
path  = r'/Users/donizetti/Doutorado/parepy' # Local path para o ParePy Toolbox
sys.path.append(path)
from parepy_toolbox import *

# def obj(x):
#     return [x[0] - x[1]]

def obj(x):
    return [12.5 * x[0] ** 3 - x[1]]

#def obj(x):    
#    return [(50 * (0.6 - x[0])**2 / (0.1 + x[1])) - 275]  

d = {'type': 'normal', 'parameters': {'mean': 0, 'std': 0.01}}
l = {'type': 'normal', 'parameters': {'mean': 0, 'std': 0.03}}
var = [d, l]

d_n = {'type': 'normal', 'parameters': {'mean': 0, 'std': 0.02}}
l_n = {'type': 'normal', 'parameters': {'mean': 0, 'std': 0.06}}
var_is = [d_n, l_n]

df, pf, beta = sampling_algorithm_structural_analysis(obj, var, 'mcs', 10000, 1, parallel=False, verbose=False, random_var_settings_importance_sampling=var_is)
print(pf)
print(beta)
print(df.head())

percentual = (df['I_0'] == 1).sum() / df.shape[0] * 100
print(f"{percentual:.2f}%")

num = 50 * (0.6 - df['X_0'])**2
denum = 0.1 + df['X_1']
print(((num / denum) > 275).sum()/10000 * 100, "%")


