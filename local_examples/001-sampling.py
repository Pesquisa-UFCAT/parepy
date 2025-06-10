import sys
path  = r'C:\git-projetos\parepy'  # PC notebook wander
sys.path.append(path)
from parepy_toolbox import *

def obj(x):
    return [12.5 * x[0] ** 3 - x[1]]

d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
var = [d, l]
df, pf, beta = sampling_algorithm_structural_analysis(obj, var, 'lhs', 1000, 1, parallel=False, verbose=False)
print(pf)
print(beta)
print(df.head())