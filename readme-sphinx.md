# Commands

- execute in terminal `sphinx-quickstart`  
- Fill admin informatation 
-  source > config.py and add extra code
```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../parepy_toolbox'))
```
- activate extension in source > config.py and add extra code  

```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

add_module_names = False
autodoc_typehints = "description"
```

- run in terminal `sphinx-apidoc -o source parepy_toolbox`

- add name folder in index.rst
```
.. toctree::
   :maxdepth: 2
   :caption: content:

   parepy_toolbox <<<< add this
```
  
- execute in terminal `sphinx-build -b html source build/html`

