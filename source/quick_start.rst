Quick Start
===========

.. raw:: html

   <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
   <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Install
-------

To use the framework in a **Python** environment, use the following command:

.. code-block:: bash

   pip install parepy-toolbox
   # or pip install --upgrade parepy-toolbox

Files structure
---------------

Let's use the example of building a problem in PAREpy using a Jupyter Notebook or **Python** file. Therefore, the basic file structure that you must assemble to use the library must be as follows:

.. code-block:: bash

   .
   .
   .
   └── problem_directory
         └── of_file.py
         └── your_problem.ipynb  # or your_problem.py
         └── file 0
         └── file 1
         └── file 2
         ...
         └── file n-1
         └── file n

The ``of_file.py`` file should contain the problem's objective function. The ``your_problem`` file will contain the call to the main function and other settings necessary for using the algorithm.

``of_file.py``
~~~~~~~~~~~~~~

``of_file.py`` is a Python function, and the user needs to define it for PAREpy to work. It has a fixed structure that must be respected, as described below:

.. code-block:: python

   def my_function(x, none_variable):
       # add your code
       return r, s, g

Parameters:
- ``x`` (list): list of design random variables. **PAREpy automatically generates these values**.
- ``none_variable`` (None, list, float, dict, str or any): The user can define this variable and input any value when calling the main function.

Returns:
- ``r`` (list): e.g., structural capacity.
- ``s`` (list): e.g., structural demand.
- ``g`` (list): State limit function \( \mathbf{G} = \mathbf{R} - \mathbf{S} \) – **mandatory**.

.. important::

   When you assemble this function, you must maintain this standard and input order.  
   The lists ``r``, ``s``, and ``g`` must have the same size, and ``g`` must be the last returned list.

Examples
~~~~~~~~

Beck [1]_ example. State Limit Function:

.. math::

   \mathbf{G} = \mathbf{R}_d - \mathbf{D} - \mathbf{L}

Option 1:

.. code-block:: python

   def example_function(x, none_variable):
       """Beck example"""
       r_d = x[0]
       d = x[1]
       l = x[2]
       r = r_d
       s = d + l
       g = r - s
       return [r], [s], [g]

Option 2:

.. code-block:: python

   def example_function(x, none_variable):
       """Beck example"""
       r_d = x[0]
       d = x[1]
       l = x[2]
       g = r_d - d - l
       return [r_d], [d + l], [g]

Option 3:

.. code-block:: python

   def example_function(x, none_variable):
       """Beck example"""
       r_d = x[0]
       d = x[1]
       l = x[2]
       r = [r_d]
       s = [d + l]
       g = [r[0] - s[0]]
       return r, s, g

Multiple State Limit Functions:

.. math::

   \mathbf{G}_0 = \mathbf{R}_d - \mathbf{D} - \mathbf{L}

.. math::

   \mathbf{G}_1 = \sigma_y \cdot W - M

.. code-block:: python

   def example_function(x, none_variable):
       """Beck example with two G"""
       r_d = x[0]
       d = x[1]
       l = x[2]
       sigma_y = x[3]
       w = x[4]
       m = x[5]

       r_0 = r_d
       s_0 = d + l
       g_0 = r_0 - s_0

       r_1 = sigma_y * w
       s_1 = m
       g_1 = r_1 - s_1

       return [r_0, r_1], [s_0, s_1], [g_0, g_1]

See more details in the following sections and verify how PAREpy can be used in your problem.

References
----------

.. [1] Beck AT. *Confiabilidade e segurança das estruturas*. Elsevier; 2019.  
       ISBN: 978-85-352-8895-7. https://www.amazon.com.br/dp/8535286888
