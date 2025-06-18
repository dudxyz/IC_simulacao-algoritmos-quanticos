
# IC\_simulacao-algoritmos-quanticos

Este repositório contém implementações em CUDA e Python para a simulação de operações fundamentais em algoritmos quânticos. O projeto foca em otimizar computações como multiplicação de vetores e matrizes complexas, além de demonstrar a aplicação do **produto de Kronecker**, essencial em simulações de sistemas quânticos compostos.

## Conteúdo do Repositório

### Soma de Vetores em CUDA

Implementação da soma de dois vetores complexos utilizando programação paralela com CUDA.

### Multiplicação Genérica de Matriz por Vetor em CUDA

Código CUDA para multiplicar matrizes (incluindo complexas) por vetores, com suporte a formatos binários de entrada e saída.

### Produto de Kronecker em CUDA

Implementação do produto de Kronecker entre matrizes complexas — operação fundamental para simular sistemas de múltiplos qubits.

### Integração Python ↔ CUDA

Sistema de ponte entre Python e CUDA para simular circuitos quânticos. A integração é feita por meio da biblioteca **Qiskit**, que gera a matriz unitária e o vetor de estado a partir de um circuito quântico, salva os dados em arquivos binários e os envia para um executável CUDA para simulação acelerada por GPU.

---

## Estrutura do Projeto

```
.
├── qiskit_env/             # Ambiente virtual Python (criado localmente)
├── ponte.py               # Script Python que orquestra a integração
├── cuda_exec.cu           # Código-fonte CUDA
├── matrix.bin             # Matriz unitária (gerado pelo Python)
├── vector.bin             # Vetor de estado inicial (gerado pelo Python)
└── result.bin             # Resultado final (gerado pelo CUDA)
```

---

##  Requisitos

* Python 3.8+
* CUDA Toolkit (com suporte ao `nvcc`)
* Qiskit
* Numpy
* Flask
* Git (opcional)

---

## Instruções de Uso

### 1. Criar Ambiente Virtual

```bash
python -m venv qiskit_env
source qiskit_env/bin/activate   # Linux/macOS
# .\qiskit_env\Scripts\activate  # Windows
```

Instale as dependências:

```bash
pip install numpy flask requests qiskit
```

---

### 2. Compilar o Executável CUDA

```bash
nvcc cuda_exec.cu -o cuda_exec   # Linux/macOS
# nvcc cuda_exec.cu -o cuda_exec.exe  # Windows
```

---

### 3. Rodar o Servidor Flask

Crie um arquivo `server.py` com o seguinte conteúdo:

<details>
<summary> Clique aqui para expandir</summary>

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/readqobj', methods=['GET'])
def read_qobj_mock():
    qobj_data = {
        "experiments": [
            {
                "instructions": [
                    {"name": "h", "qubits": [0]},
                    {"name": "cx", "qubits": [0, 1]}
                ],
                "n_qubits": 2
            }
        ]
    }
    return jsonify(qobj_data)

@app.route('/', methods=['POST'])
def receive_result():
    data = request.get_json()
    print("Resultado recebido:", data)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

</details>

Execute o servidor:

```bash
python server.py
```

---

### 4. Executar o Script Principal

Com o servidor rodando e o ambiente virtual ativado:

```bash
python ponte.py
```

---

## Sobre a Aplicação

Este projeto surgiu no contexto de uma Iniciação Científica (IC) voltada para **simulação de algoritmos quânticos em GPU**. A integração CUDA + Python permite acelerar o processo de multiplicação de matrizes complexas e aplicação de circuitos quânticos simulados localmente, com objetivo de estudar alternativas ao backend padrão do Qiskit.


