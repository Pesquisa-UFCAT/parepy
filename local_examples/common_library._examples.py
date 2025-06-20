import sys
path = r'C:\git-projetos\parepy'  # PC notebook wander
sys.path.append(path)
from parepy_toolbox import *

# pip install -U parepy-toolbox  # Use this command if you need install or update this framework
from parepy_toolbox import pf_equation
beta = 3.5
pf = pf_equation(beta)
print(f"Probability of failure: {pf:.5e}")

# pip install -U parepy-toolbox
from parepy_toolbox import beta_equation
pf = 2.32629e-04
beta = beta_equation(pf)
print(f"Reliability index: {beta:.5f}")

# pip install -U parepy-toolbox
from parepy_toolbox import sampling_kernel_without_time
def obj(x): # We reccomend to create this py function in other .py file when use parellel process and .ipynb code
        g_0 = 12.5 * x[0] ** 3 - x[1]
        return [g_0]
d = {'type': 'normal', 'parameters': {'mean': 1.0, 'std': 0.1}}
l = {'type': 'normal', 'parameters': {'mean': 10.0, 'std': 1.0}}
var = [d, l]
number_of_limit_functions = 1
method = 'mcs'
n_samples = 10000
start = time.perf_counter()
df = sampling_kernel_without_time(obj, var, method, n_samples, number_of_limit_functions)
end = time.perf_counter()
print(f"Time elapsed: {(end-start):.5f} s")
print(df)