# How to create a virtual environment

All commands needed to type in VS terminal:
  
- Create the Virtual Environment: `python -m venv env_name`. *The standard pattern for virtual environments in `.gitignore` is `env`*.      
- In **Windows**: `env_name\Scripts\activate`. In **macOS/Linux**: `source env_name/bin/activate`.     
- The terminal prompt will show `(env_name) $` to indicate activation. 
- Install project dependencies: `pip install -r requirements.txt`.  
- When done, exit the environment with: `deactivate`.  