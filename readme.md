# Biblioteca PAREpy

Essa é a biblioteca **PAREpy** para simulação e aplicação de engenharia probabilística. 
   
Esse é a documentação para desenvolvedores que pretendem utilizar essa biblioteca como ferramenta de pesquisa. A seguir serão abordados dos tópicos necessários para rodar os exemplos dessa biblioteca em seu computador seja ele Linux, Mac ou Windows.

# Requerimentos para desenvolvimento

- Python  
- Biblioteca Poetry

Caso não tenha o **Poetry** execute:

```python
pip install poetry # ou pip install --upgrade poetry 
```

# Comandos básicos Poetry

Para executar o programa sem necessidade de instalação de todas as biblioteca utilize o gerenciador Poetry, com os seguintes comandos:

```python
colocar aqui sequência de comandos e instruções se precisar figuras faz um gifezinho aqui para o pessoal ver vc mexendo
```

# O que é o arquivo ```common_library.py```

É um arquivo que contém as bibliotecas comuns a todos os métodos de engenharia probabilística utilizados para construir os algoritmos.

# O que é o ```arquivo pare.py```

É o arquivo que contém os principais algoritmos da biblioteca e que normalmente são empregados pelos usuários.


# Configuração com Poetry

Este tutorial irá guiá-lo pelos passos básicos para configurar seu ambiente de desenvolvimento utilizando o Poetry, instalar as bibliotecas necessárias e garantir que você está utilizando o ambiente virtual correto para rodar seus notebooks.

## Passo 1: Ativar a Venv com o Poetry

Para ativar a venv criada pelo Poetry, siga estes passos:

1. Navegue até o diretório do seu projeto onde o arquivo `pyproject.toml` está localizado, no nosso caso, está dentro da pasta principal.

   ```bash
   cd .../PAREPYDEV/
   ```

2. Ative o ambiente virtual criado pelo Poetry com o seguinte comando:

   ```bash
   poetry shell
   ```

   Este comando ativa o ambiente virtual e você verá o prompt do terminal mudar para indicar que o ambiente virtual está ativo.

## Passo 2: Instalar as Bibliotecas com o Poetry

Com o ambiente virtual ativado, instale as bibliotecas necessárias usando o Poetry:

```bash
poetry install
```

Este comando irá instalar todas as dependências listadas no arquivo `pyproject.toml`.

## Passo 3: Selecionar a Venv no Jupyter Notebook

Se você estiver utilizando Jupyter Notebook, é essencial garantir que o ambiente virtual correto esteja selecionado. Siga estes passos:

1. Abra o Jupyter Notebook.

2. Verifique se o kernel do notebook está configurado para usar o ambiente virtual criado pelo Poetry, normalmente ele se encontra no canto superior direito. Se você não vê o ambiente virtual na lista de kernels, adicione-o com o seguinte comando:

   ```bash
   poetry run ipython kernel install --user --name=<nome-do-seu-ambiente> --display-name "Python (venv)"
   ```

   Substitua `<nome-do-seu-ambiente>` pelo nome que você deseja para o ambiente virtual.

3. Ao criar um novo notebook ou abrir um existente, selecione o kernel correspondente ao seu ambiente virtual.

Seguindo estes passos, você garantirá que está trabalhando no ambiente virtual correto e utilizando as dependências necessárias para seu projeto.

Caso precise de mais ajuda, consulte a [documentação do Poetry](https://python-poetry.org/docs/) e a [documentação do Jupyter](https://jupyter.org/documentation) para mais detalhes.
