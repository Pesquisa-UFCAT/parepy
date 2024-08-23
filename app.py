import streamlit as st

def generate_function_code(name, num_vars, body):
    """
    Generates Python code for a function.

    Args:
        name (str): Name of the function.
        num_vars (int): Number of input variables.
        body (str): Body of the function, which should start with "g =".

    Returns:
        str: Python function code with the body and return fixed as "g".
    """

    vars_str = ', '.join([f'var{i+1}' for i in range(num_vars)])
    return f"""def {name}({vars_str}):
    {body}
    return g
"""

st.title('Gerador de Código Python')

func_name = st.text_input('Nome da função:')
num_vars = st.number_input('Número de variáveis:', min_value=0, step=1)
func_body = st.text_area('Corpo da função:', help='Certifique-se de que o corpo da função comece com "g =".')

if st.button('Gerar Código'):
    if func_name and func_body:
        if not func_body.strip().startswith('g ='):
            st.error('O corpo da função deve começar com "g =".')
        else:
            code = generate_function_code(func_name, num_vars, func_body)
            
            st.code(code, language='python')
            
            with open('generated_function.py', 'w') as file:
                file.write(code)
            
            with open('generated_function.py', 'rb') as file:
                st.download_button(
                    label='Baixar arquivo .py',
                    data=file,
                    file_name='generated_function.py'
                )
    else:
        st.error('Por favor, preencha todos os campos necessários.')
