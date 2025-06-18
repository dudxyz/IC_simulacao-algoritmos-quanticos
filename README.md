# IC_simulacao-algoritmos-quanticos

## Soma de vetores em CUDA
...

## Multiplicação genérica de matriz por vetor em CUDA
...

## Produto kronecker de duas matrizes complexas em CUDA
...

## Ponte de Integração Python-CUDA 
Este parte do repositório se refere a um projeto que demonstra a integração entre um script Python e um executável CUDA para realizar operações de computação quântica (multiplicação de matriz-vetor) de forma acelerada por GPU.

O script ponte.py atua como uma "ponte", recebendo uma descrição de circuito quântico em formato JSON (antigo Qobj), construindo um QuantumCircuit no Qiskit, gerando a matriz unitária e o vetor de estado correspondentes, salvando-os em arquivos binários, executando um programa CUDA (cuda_exec) para realizar a computação e, finalmente, lendo o resultado e enviando-o de volta via HTTP POST.

Pré-requisitos
Antes de começar, certifique-se de ter o seguinte instalado:

Python 3.8+

Pip (gerenciador de pacotes do Python)

CUDA Toolkit (incluindo nvcc)

Git (opcional, para clonar o repositório)

Estrutura do Projeto
.
├── qiskit_env/             # Ambiente virtual Python (será criado)
├── ponte.py               # Script Python (a "ponte")
├── cuda_exec.cu           # Código-fonte CUDA
├── matrix.bin             # Arquivo de saída da matriz (gerado pelo Python)
├── vector.bin             # Arquivo de saída do vetor (gerado pelo Python)
└── result.bin             # Arquivo de saída do resultado CUDA (gerado pelo CUDA)

Configuração e Execução
Siga os passos abaixo para configurar e executar o projeto:

1. Configurar o Ambiente Virtual Python
É altamente recomendável usar um ambiente virtual para isolar as dependências do projeto.

Navegue até o diretório raiz do projeto no seu terminal:

cd /caminho/para/sua/pasta/do/projeto

Crie um novo ambiente virtual:

python -m venv qiskit_env

Ative o ambiente virtual:

No Windows:

.\qiskit_env\Scripts\activate

No macOS/Linux:

source qiskit_env/bin/activate

Você verá (qiskit_env) no início da sua linha de comando, indicando que o ambiente está ativo.

Instale as dependências Python dentro do ambiente ativado:

pip install numpy requests qiskit

2. Compilar o Código CUDA
O código-fonte CUDA (cuda_exec.cu) precisa ser compilado para se tornar um executável.

Certifique-se de que o ambiente virtual está ativo (o (qiskit_env) deve aparecer no seu terminal).

Navegue até o diretório onde cuda_exec.cu está localizado (geralmente o mesmo diretório de ponte.py).

Compile o arquivo .cu usando nvcc:

No Windows:

nvcc cuda_exec.cu -o cuda_exec.exe

No macOS/Linux:

nvcc cuda_exec.cu -o cuda_exec

Se a compilação for bem-sucedida, um arquivo executável chamado cuda_exec (ou cuda_exec.exe no Windows) será criado no mesmo diretório.

(Apenas macOS/Linux) Conceda permissões de execução:

chmod +x cuda_exec

3. Executar o Servidor Flask (Necessário)
O script ponte.py faz uma requisição GET para http://127.0.0.1:5000/readqobj para obter o JSON do circuito e uma requisição POST para http://127.0.0.1:5000/ para enviar o resultado. Você precisa ter um servidor Flask (ou similar) rodando nessas rotas.

Exemplo básico de um server.py para testes:

# server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

# Rota GET para simular a leitura do Qobj
@app.route('/readqobj', methods=['GET'])
def read_qobj_mock():
    # Este é um Qobj de exemplo. Na realidade, você leria de um arquivo ou geraria dinamicamente.
    qobj_data = {
        "experiments": [
            {
                "clbit_labels": [],
                "creg_sizes": [],
                "instructions": [
                    {"name": "h", "qubits": [0]},
                    {"name": "cx", "qubits": [0, 1]}
                ],
                "memory_slots": 0,
                "n_qubits": 2,
                "name": "default_experiment",
                "qreg_sizes": [],
                "qubit_labels": [0, 1]
            }
        ],
        "header": {"backend_name": "unknown", "backend_version": "unknown"},
        "qobj_data": {
            "backend_name": "unknown", "backend_version": "unknown",
            "init_qubits": None, "meas_level": "none", "memory": False,
            "memory_slots": 0, "n_qubits": 2, "parameter_binds": None,
            "parametric_pulses": None, "shots": 1024
        }
    }
    return jsonify(qobj_data)

# Rota POST para receber os resultados
@app.route('/', methods=['POST'])
def receive_result():
    if request.is_json:
        data = request.get_json()
        print("\n--- Resultado Recebido via POST ---")
        print("Dados recebidos (formato complexo serializado):", data)
        # Opcional: Converter de volta para números complexos se necessário
        # complex_results = [complex(item['real'], item['imag']) for item in data]
        # print("Resultados convertidos para complexo:", complex_results)
        print("-----------------------------------")
        return jsonify({"status": "success", "message": "Resultado recebido com sucesso!"}), 200
    else:
        return jsonify({"status": "error", "message": "Requisição deve ser JSON"}), 400

if __name__ == '__main__':
    print("Iniciando servidor Flask em http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)


Salve o código acima como server.py no mesmo diretório de ponte.py.

Instale Flask no seu ambiente virtual (se ainda não tiver):

pip install Flask

Inicie o servidor Flask em um terminal separado (mantenha-o rodando):

python server.py

4. Executar o Script Python (ponte.py)
Com o ambiente ativado e o executável CUDA compilado (e o servidor Flask rodando), você pode executar o script Python.

Certifique-se de que o ambiente virtual está ativo no terminal onde você irá executar ponte.py.

Execute o script:

python ponte.py

Você verá a saída do ponte.py no seu terminal, incluindo o circuito, a confirmação de que os arquivos binários foram salvos, a execução do CUDA e o resultado lido. No terminal onde o server.py está rodando, você deverá ver a mensagem indicando que o resultado foi recebido via POST.
