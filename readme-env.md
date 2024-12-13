# Criando um ambiente virtual

Este é um guia detalhado para a criação e utilização de um ambiente virtual Python com o uso do `venv`, um recurso integrado ao Python para a criação de ambientes autônomos. Isso é benéfico para administrar as dependências de projetos de maneira autônoma e prevenir conflitos de pacotes entre projetos distintos. 

## Passo 1: Verificar se o Python está instalado
Antes de criar um ambiente virtual, é importante garantir que você tem o Python instalado.

No terminal (ou prompt de comando), execute:

```python
python --version
```

Se o Python estiver instalado, isso mostrará a versão. Certifique-se de que seja superior ao Python 3.x.

## Passo 2: Criar o ambiente virtual
Com o Python instalado, agora você pode criar um ambiente virtual. Use o comando:

```python
python -m venv nome_do_ambiente
```

Aqui, `nome_do_ambiente` é o nome que você deseja dar ao seu ambiente virtual. Por exemplo, pode ser `meuambiente`.

## Passo 3: Ativar o ambiente virtual
Uma vez criado o ambiente virtual, é necessário ativá-lo. O método de ativação depende do seu sistema operacional:

* **No Windows**:
```bash
nome_do_ambiente\Scripts\activate
```

* **No macOS/Linux**:
```bash
source nome_do_ambiente/bin/activate
```

Após a ativação, você verá que o nome do ambiente virtual aparecerá antes do prompt do terminal, indicando que ele está ativo, algo como:

```bash
(meuambiente) $
```

## Passo 4: Instalar pacotes no ambiente virtual
Com o ambiente ativado, qualquer pacote Python instalado via `pip` será isolado nesse ambiente virtual. Por exemplo, para instalar o pacote `numpy`, execute:

```python
pip install numpy
```

Para instalar empregue o arquivo `requirements.txt`. Basta executar o comando python. Na falta de qualquer pacote use o comando empregado anteriormente.

```python
pip install -r requirements.txt
```

## Passo 5: Desativar o ambiente virtual
Quando terminar de trabalhar, você pode desativar o ambiente virtual com o comando:

```bash
deactivate
```

Após desativar, o prompt do terminal voltará ao normal e você estará fora do ambiente virtual.

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
