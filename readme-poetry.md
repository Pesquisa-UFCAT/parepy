### Tutorial 2: Como Instalar e Usar o Poetry para Gerenciar Ambientes Virtuais

Este é um guia detalhado para a criação e utilização do gerenciador de dependências e ambientes virtuais chamado **Poetry**, facilitando a instalação de pacotes e o controle de versões, garantindo ambientes isolados para cada projeto.

---

### 1. Instalação do Poetry

#### Requisitos:
- **Python 3.7+** instalado em seu sistema.

#### Passos:

1. **Instalar o Poetry**
   
   No terminal, use o comando a seguir para instalar o Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Ou, alternativamente, com `pip`:
   ```bash
   pip install poetry
   ```

2. **Verifique a instalação**
   
   Após a instalação, você pode verificar se o Poetry foi instalado corretamente executando:
   ```bash
   poetry --version
   ```

---

### 2. Criando um Ambiente Virtual com o Poetry

Com o Poetry instalado, siga os passos abaixo para configurar o ambiente virtual.

1. **Inicie um novo projeto**
   
   Na pasta onde deseja criar seu projeto, execute o seguinte comando:
   ```bash
   poetry new nome_do_projeto
   ```

   Isso criará uma estrutura de diretório com `pyproject.toml` e um diretório `nome_do_projeto` contendo o arquivo `__init__.py`.

   Se já tiver um projeto existente, você pode apenas executar o comando dentro da pasta:
   ```bash
   poetry init
   ```
   Isso inicializa o projeto sem criar uma nova estrutura de pastas.

2. **Instale as dependências e crie o ambiente virtual**
   
   Para instalar as dependências listadas no arquivo `pyproject.toml` e criar o ambiente virtual automaticamente, execute:
   ```bash
   poetry install
   ```

   O Poetry criará um ambiente virtual isolado e instalará as dependências.

---

### 3. Ativando o Ambiente Virtual

O Poetry cuida do gerenciamento do ambiente virtual automaticamente, mas se você quiser ativá-lo manualmente, siga os passos:

1. **Ativando o ambiente virtual manualmente**
   
   Para ativar o ambiente virtual criado pelo Poetry, você pode usar o comando:
   ```bash
   poetry shell
   ```
   
   Isso abrirá um novo shell com o ambiente virtual ativado.

2. **Executando comandos dentro do ambiente virtual sem ativar o shell**
   
   Se preferir executar comandos dentro do ambiente virtual sem ativá-lo explicitamente, use:
   ```bash
   poetry run python script.py
   ```

---

### 4. Desativando o Ambiente Virtual

Para desativar o ambiente virtual e sair do shell, basta usar o comando:
```bash
exit
```
Isso encerrará o shell do ambiente virtual e o levará de volta ao seu shell padrão.

---

### 5. Deletando o Ambiente Virtual

Caso você queira deletar o ambiente virtual criado pelo Poetry, siga os passos abaixo:

1. **Remova o ambiente virtual**
   
   Para excluir o ambiente virtual, use o comando:
   ```bash
   poetry env remove python
   ```

   Esse comando remove o ambiente virtual associado ao seu projeto.

---

### 6. Outras Operações Úteis

- **Verificar o status do ambiente virtual**:
   ```bash
   poetry env list
   ```
- **Recriar o ambiente virtual** (caso tenha deletado ou queira resetar):
   ```bash
   poetry install
   ```

### 7. Publicando o pacote

- **Build framework**:
   ```bash
   poetry build
   ```
- **Publish framework**
   ```bash
   poetry publish
   ```