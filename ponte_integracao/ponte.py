# ponte.py
# Ponte Python: Recebe Qobj, converte para matriz e vetor, roda CUDA e retorna resultado

import numpy as np
import requests
import json
import subprocess
from qiskit.qobj import Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator

#  requisição GET para obter Qobj JSON
url = "http://127.0.0.1:5000"  # <- alterar ip
res = requests.get(url)
qobj_json = res.json()

# converte Qobj para circuito
qobj = Qobj.from_dict(qobj_json)
circ = qobj.to_circuit()

# gera matriz unitária e vetor de estado
unitary = circ.to_operator().data.astype(np.complex64)
state = Statevector.from_instruction(circ).data.astype(np.complex64)

# salva como binário
unitary.tofile("matrix.bin")
state.tofile("vector.bin")

# executa o binário CUDA
subprocess.run(["./cuda_exec", str(unitary.shape[0])])  # passa a dimensão

# lê resultado e envia de volta
result = np.fromfile("result.bin", dtype=np.complex64)
print("Resultado:", result)

# enviar resultado de volta via POST
requests.post("http://127.0.0.1:5000", json=result.tolist())
