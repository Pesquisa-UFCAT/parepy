import sys
path = r'C:\git-projetos\parepy'  # PC notebook wander
path = r'D:\py\parepy'
sys.path.append(path)
from parepy_toolbox import *

def obj(x):
    return [12.5 * x[0] ** 3 - x[1], 12.5 * x[0] ** 3 - x[1]]

d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
var = [d, l]
df, pf, beta = sampling_algorithm_structural_analysis(obj, var, 'lhs', 1000, 2, parallel=False, verbose=False)
print(pf)
print(beta['beta_0'].iloc[0])
print(df.head())