
def obj_function(x, none_variable):
    # Random variables
    f_y = x[0]
    p_load = x[1]
    w_load = x[2]

    capacity = 80 * x[0]
    demand = 54 * x[1] + 5832 * x[2]

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]
