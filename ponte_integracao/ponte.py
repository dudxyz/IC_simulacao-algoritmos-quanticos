# ponte.py
# Ponte Python: Recebe Qobj JSON (formato antigo), converte para QuantumCircuit,
# gera matriz e vetor, roda CUDA e retorna resultado.
import numpy as np
import requests
import json
import subprocess
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

# requisição GET para obter Qobj JSON
url = "http://127.0.0.1:5000/readqobj?namePath=circuit.qobj"  # <- ip helena
try:
    res = requests.get(url)
    res.raise_for_status()
    qobj_json = res.json()
except requests.exceptions.RequestException as e:
    print(f"Erro ao fazer a requisição HTTP: {e}")
    print("Verifique se o servidor Flask está rodando e o IP/porta estão corretos.")
    exit()

# converte Qobj JSON (formato antigo) para QuantumCircuit (Qiskit 1.0+) (obs: env)
try:
    experiment_data = qobj_json["experiments"][0]
    n_qubits = experiment_data["n_qubits"]
    num_clbits = experiment_data.get("memory_slots", 0)

    circ = QuantumCircuit(n_qubits, num_clbits)

    for instruction in experiment_data["instructions"]:
        name = instruction["name"]
        qubits = instruction["qubits"]

        if name == "h":
            circ.h(qubits[0])
        elif name == "cx":
            circ.cx(qubits[0], qubits[1])
        elif name == "measure":
            clbits = instruction["clbits"]
            for q, c in zip(qubits, clbits):
                circ.measure(q, c)
        else:
            print(f"Aviso: Porta '{name}' não reconhecida e ignorada.")

except KeyError as e:
    print(f"Erro de chave no JSON do Qobj: {e}")
    print("O formato do Qobj JSON pode não ser o esperado. Verifique a estrutura.")
    exit()
except IndexError:
    print("Erro: 'experiments' vazio ou não encontrado no JSON do Qobj.")
    exit()
except Exception as e:
    print(
        f"Erro inesperado ao parsear o Qobj JSON ou construir o circuito: {e}")
    exit()

print("Circuito quântico construído a partir do JSON:")
print(circ.draw())

# gera matriz unitária e vetor de estado
try:
    unitary = Operator(circ).data.astype(np.complex64)
    state = Statevector.from_instruction(circ).data.astype(np.complex64)
except Exception as e:
    print(f"Erro ao gerar matriz unitária ou vetor de estado: {e}")
    print("Isso pode ocorrer se o circuito for muito complexo para essas operações ou se houver outro problema.")
    exit()

# salva como binário
try:
    unitary.tofile("matrix.bin")
    state.tofile("vector.bin")
    print("Matriz unitária e vetor de estado salvos como 'matrix.bin' e 'vector.bin'.")
except IOError as e:
    print(f"Erro ao salvar arquivos binários: {e}")
    exit()

# executa o binário CUDA
try:
    print(f"Executando o binário CUDA com dimensão: {unitary.shape[0]}")
    cuda_process = subprocess.run(
        ["./cuda_exec", str(unitary.shape[0])], check=True)
except FileNotFoundError:
    print("Erro: 'cuda_exec' não encontrado. Verifique o caminho ou se o executável foi compilado.")
    exit()
except subprocess.CalledProcessError as e:
    print(f"Erro na execução do binário CUDA: {e}")
    print(f"Saída do stderr: {e.stderr.decode()}")
    exit()
except Exception as e:
    print(f"Erro inesperado ao executar o binário CUDA: {e}")
    exit()

# lê resultado e envia de volta
try:
    result = np.fromfile("result.bin", dtype=np.complex64)
    print("Resultado lido de 'result.bin':", result)

    # CORREÇÃO AQUI: converter números complexos para um formato serializável JSON
    # convertemos cada número complexo para um dicionário {'real': parte_real, 'imag': parte_imaginaria}
    serializable_result = []
    for val in result:
        serializable_result.append(
            {'real': float(val.real), 'imag': float(val.imag)})

    # enviar resultado de volta via POST (ainda não recebíveis no servidor)
    post_res = requests.post("http://127.0.0.1:5000", json=serializable_result)
    post_res.raise_for_status()
    print("Resultado enviado com sucesso via POST para http://127.0.0.1:5000.")
except FileNotFoundError:
    print("Erro: 'result.bin' não encontrado. O executável CUDA pode não ter gerado o arquivo.")
except requests.exceptions.RequestException as e:
    print(f"Erro ao enviar o resultado via POST: {e}")
except Exception as e:
    print(f"Erro inesperado ao ler ou enviar o resultado: {e}")
