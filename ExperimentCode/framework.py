from qiskit import QuantumCircuit, QuantumRegister, Aer, opflow
from qiskit.circuit.library import Diagonal
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.extensions import UnitaryGate
import matplotlib.pyplot as plt
import numpy as np


def count_bits(n):
    count = 0
    while n > 0:
        count += 1
        n >>= 1
    return count


def nearest_power_of_two(n):
    power = 1
    while power < n:
        power <<= 1
    return power


class Framework(object):
    def __init__(self):
        self.m = 0
        self.n = 0
        self.arrayGate = None
        self.operationGate = None
        self.ansatz = None
        self.data = None
        pass

    def __bitWeightGate(self, kn: int):
        qc = QuantumCircuit(kn)
        for i in range(kn):
            qc.ry(np.arctan(1.0 / (1 << (1 << i))) * 2, i)
        return qc

    def encode(self, arr: np.ndarray):
        assert (arr.size & (arr.size - 1)
                ) == 0 and arr.size != 0, "数组长度不是 2 的幂次方"
        self.n = arr.size
        self.m = nearest_power_of_two(count_bits(np.max(arr)))

        data = []
        for i in arr:
            data.append(i)
            data.append((1 << self.m) - 1)
        data = [np.binary_repr(x, self.m) for x in data]
        data = np.array([[int(i) for i in x] for x in data])
        self.data = data
        data = data.flatten()
        data[data == 0] = -1
        self.arrayGate = Diagonal(data)

    def operation(self, o: np.ndarray):
        assert o.size == self.n, "求和长度和数组长度不匹配"

        x = np.mat(np.zeros(self.n)).T
        x[0] = 1
        o = o / np.linalg.norm(o)
        o = np.mat(o).T
        O = (x+o)*((x+o).T)/(1+x.T*o)-np.identity(self.n)
        self.operationGate = UnitaryGate(O)

    def __observable(self, n: int) -> opflow.OperatorBase:
        obs0 = (opflow.Z + opflow.I) / 2
        return obs0.tensorpower(n)

    def set_ansatz(self, ansatz):
        self.ansatz = ansatz

    def loss_qc(self):
        km = int(np.log2(self.m))
        kn = int(np.log2(self.n))

        qubit_j = QuantumRegister(km, 'j')
        qubit_aux = QuantumRegister(1, 'aux')
        qubit_i = QuantumRegister(kn, 'i')

        circ = QuantumCircuit(qubit_j, qubit_aux, qubit_i)

        bwGate = self.__bitWeightGate(km)
        circ.compose(bwGate.to_gate(label='BitWeight'),
                     qubits=qubit_j, inplace=True)
        circ.h(qubit_aux)
        circ.h(qubit_i)

        circ.compose(self.arrayGate.to_gate(label='E(array)'),
                     qubits=qubit_j[:]+qubit_aux[:]+qubit_i[:], inplace=True)

        circ.compose(self.ansatz, inplace=True, qubits=qubit_i)

        circ.h(qubit_j)
        circ.h(qubit_aux)
        circ.compose(self.operationGate.inverse(),
                     qubits=qubit_i[:], inplace=True)

        return circ

    def Run(self, seed, init_point, max=False):
        qc = self.loss_qc()

        algorithm_globals.random_seed = seed

        qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                             seed_transpiler=seed, seed_simulator=seed,
                             shots=1024)
        optim = SLSQP(maxiter=1000)

        vqe = VQE(qc, quantum_instance=qi, initial_point=init_point,
                  optimizer=optim)

        target = self.__observable(qc.num_qubits)
        if max == True:
            target = -target
        obj = vqe.compute_minimum_eigenvalue(target)

        return obj
