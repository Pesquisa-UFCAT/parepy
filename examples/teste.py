import numpy as np
import time
from multiprocessing import Pool
from old.obj_function import nowak_example_time, nowak_example, evaluation_model_parallel



if __name__ == '__main__':
    # Selecting a sample to test
    n_constraints = 1
    none_variable = None
    obj = nowak_example
    dataset_x = np.random.rand(21, 3)
    parts = np.array_split(dataset_x, 5)

    capacity = np.zeros((len(dataset_x), n_constraints))
    demand = np.zeros((len(dataset_x), n_constraints))
    state_limit = np.zeros((len(dataset_x), n_constraints))
    indicator_function = np.zeros((len(dataset_x), n_constraints))

    # Parallel test
    start_time_parallel = time.time()
    capacity = np.zeros((len(dataset_x), n_constraints))
    demand = np.zeros((len(dataset_x), n_constraints))
    state_limit = np.zeros((len(dataset_x), n_constraints))
    indicator_function = np.zeros((len(dataset_x), n_constraints))
    information_model = [[i, obj, none_variable] for i in parts]
    with Pool() as p:
        result = p.map_async(func=evaluation_model_parallel, iterable=information_model)
        result = result.get()
    cont = 0
    for i in range(len(parts)):
        for j in range(tam[i]):
            capacity[cont, :] = result[i][0][j].copy()
            demand[cont, :] = result[i][1][j].copy()
            state_limit[cont, :] = result[i][2][j].copy()
            indicator_function[cont, :] = [0 if value <= 0 else 1 for value in result[i][2][j]]
            cont += 1

    print('Parallel time: ', time.time() - start_time_parallel)