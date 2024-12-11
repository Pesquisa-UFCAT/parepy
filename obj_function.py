"""Objective function file for the probabilistic problem."""

def nowak_collins_example(x, none_variable):
    """Objective function for the Nowak example (tutorial).
    """

    # Random variables
    f_y = x[0]
    p_load = x[1]
    w_load = x[2]
    capacity = 80 * f_y
    demand = 54 * p_load + 5832 * w_load

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]


def nowak_collins_time_example(x, none_variable):
    """Objective function for the Nowak example (tutorial).
    """
    
    # User must copy and paste this code in time reliability objective function
    ###########################################
    id_analysis = int(x[-1])
    time_step = none_variable['time analysis']
    t_i = time_step[id_analysis]
    print(t_i)
    # t_i is a time value from your list of times entered in the 'none variable' key.
    ###########################################

    # Random variables
    f_y = x[0]
    p_load = x[1]
    w_load = x[2]
    
    # Degradation criteria
    if t_i == 0:
        degrad = 1
    else:
        degrad = 1 - (0.2 / t_i) * 1E-2

    # Capacity and demand
    capacity = 80 * f_y * degrad
    demand = 54 * p_load + 5832 * w_load

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]


def beck_example(x, none_variable):
    y_1 = x[0]
    y_2 = x[1]
    y_3 = x[2]
    # g = 12.50 * y_1 * y_2 + 100 * y_2 + 250 * y_1 - 200 * y_3 + 1000
    g = y_1 * y_2 - y_3
    
    return g


def grad_beck_example(x, none_variable):
    y_1 = x[0]
    y_2 = x[1]
    y_3 = x[2]
    # g_grad = [12.50 * y_2 + 250, 12.50 * y_1 + 100, -200]
    g_grad = [y_2, y_1, -1]
    
    return g_grad


def beck_example_2(x, none_variable):
    r = x[0]
    d = x[1]
    l = x[2]
    g = r - (d + l)
    
    return g


def grad_beck_example_2(x, none_variable):
    r = x[0]
    d = x[1]
    l = x[2]
    g = [1, -1, -1]
    
    return g


def fosm_3(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    g = x1**3 + x2**3 - 18
    
    return g


def grad_fosm_3(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    g = [3*x1**2, 3*x2**2] 
    
    return g


def form_1(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    g = x1*x2 - 1400
    
    return g


def grad_form_1(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    g = [x2, x1] 
    
    return g