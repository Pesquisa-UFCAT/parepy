import streamlit as st

def generate_function_code(name, num_vars, body):
    """
    Generates Python code for a function.

    Args:
        name (str): Name of the function.
        num_vars (int): Number of input variables.
        body (str): Body of the function, which should use 'x_0', 'x_1', etc.
                    The function will replace these with 'x[0]', 'x[1]', etc.

    Returns:
        str: Python function code with the body and return fixed as "g".
    """
    
    # Verifica se todas as variáveis estão presentes
    for i in range(num_vars):
        if f'x_{i}' not in body:
            raise ValueError(f'O corpo da função deve conter x_{i}.')

    # Substitui as variáveis x_0, x_1, ... por x[0], x[1], ...
    for i in range(num_vars):
        body = body.replace(f'x_{i}', f'x[{i}]')

    # Gera o código da função
    return f"""def {name}(x, none_variable):
    {body}
    return g
"""

st.title('Gerador de Código Python')

st.write("""
Esta aplicação permite ao usuário gerar um código para uma função Python. 
Para isso, o usuário fornece o nome da função, o número de variáveis de entrada e o corpo da função. 
O código gerado incluirá automaticamente uma variável 'g' retornada pela função. 
As variáveis devem ser escritas no formato 'x_0', 'x_1', etc., que serão convertidas para 'x[0]', 'x[1]', etc.
Após gerar o código, o usuário poderá visualizá-lo na tela e fazer o download do arquivo Python.
""")

func_name = st.text_input('Nome da função:')
num_vars = st.number_input('Número de variáveis:', min_value=1, step=1)
func_body = st.text_area('Corpo da função:', help='Certifique-se de usar variáveis como "x_0", "x_1", etc.')

if st.button('Gerar Código'):
    if func_name and func_body:
        try:
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
        except ValueError as e:
            st.error(str(e))
    else:
        st.error('Por favor, preencha todos os campos necessários.')
