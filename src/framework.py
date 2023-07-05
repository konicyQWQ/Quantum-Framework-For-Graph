from qiskit import QuantumCircuit, QuantumRegister, opflow
from qiskit.quantum_info import Operator
from qiskit.circuit.library import Diagonal
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator
from .ansatz import s4_ansatz, sample_exact_thetas
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
        self.n = 0
        self.m = 0
        self.arrayGate = None
        self.sumGate = None
        self.data = None
        self.ansatz = None
        self.ansatz_type = None
        self.ansatz_qc = None
        self.sum_vec = None
        pass

    def __bitWeightGate(self, km: int):
        qc = QuantumCircuit(km)
        for i in range(km):
            qc.ry(np.arctan(1.0 / (1 << (1 << i))) * 2, i)
        return qc

    def __observable(self, n: int):
        obs0 = (opflow.Z + opflow.I) / 2
        return obs0.tensorpower(n)

    def encode(self, arr: np.ndarray):
        assert (arr.size & (arr.size - 1)
                ) == 0 and arr.size != 0, "size of array must be power of 2"
        self.n = arr.size
        self.m = nearest_power_of_two(count_bits(np.max(arr)))

        self.data = np.copy(arr)
        data = []
        for i in arr:
            data.append(i)
            data.append((1 << self.m) - 1)

        data = [np.binary_repr(x, self.m) for x in data]
        data = np.array([[int(i) for i in x] for x in data])
        data = data.flatten()
        data[data == 0] = -1
        self.arrayGate = Diagonal(data)

    def sum(self, o: np.ndarray):
        assert o.size == self.n, "the size of sum vector |o> should equal to n"

        self.sum_vec = o
        x = np.mat(np.zeros(self.n)).T
        x[0] = 1
        o = o / np.linalg.norm(o)
        o = np.mat(o).T
        O = (x+o)*((x+o).T)/(1+x.T*o)-np.identity(self.n)
        self.sumGate = UnitaryGate(O)

    def set_ansatz(self, ansatz):
        if ansatz == 'permutation':
            self.ansatz_type = 'permutation'
            qbit = int(np.log2(self.n))
            qc, _ = s4_ansatz('circular', qreg=qbit)
            self.ansatz = qc
            self.ansatz_qc = qc
        elif ansatz == 'vertex permutation':
            self.ansatz_type = 'vertex permutation'
            qbit = int(np.log2(self.n) / 2)
            qc, _ = s4_ansatz('circular', qreg=qbit)
            ansatz = QuantumCircuit(qbit * 2)
            ansatz.compose(qc, inplace=True, qubits=range(qbit))
            ansatz.compose(qc, inplace=True,
                           qubits=np.array(range(qbit)) + qbit)
            self.ansatz = ansatz
            self.ansatz_qc = qc
        else:
            self.ansatz = ansatz
            self.ansatz_qc = ansatz

    def __loss_qc(self):
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
        circ.compose(self.sumGate.inverse(),
                     qubits=qubit_i[:], inplace=True)

        return circ

    def run(self, max=False, seed=None, init_point=None, maxiter=100):
        qc = self.__loss_qc()
        target = self.__observable(qc.num_qubits)

        if max == True:
            target = -target

        np.random.seed(seed)
        algorithm_globals.random_seed = seed
        
        if init_point == None:
            init_point = (np.random.uniform(size=len(qc.parameters))) * np.pi

        qi = QuantumInstance(AerSimulator(method='statevector'),
                            seed_transpiler=seed, seed_simulator=seed,
                            shots=1024)
        
        answer = []
        def calc(x):
            new_qc = qc.bind_parameters(x)
            result = qi.execute(new_qc)
            return float(result.results[0].data.statevector[0].real)
        
        def callback(xk):
            ans = []
            sampled_params_dicts = sample_exact_thetas(xk, n=32, seed=np.random.randint(2147483647))
            if self.ansatz_type == 'permutation':
                for v in sampled_params_dicts:
                    p = np.abs(Operator(self.ansatz_qc.bind_parameters(v)).data)
                    p = np.round(p)
                    cost = (self.sum_vec @ p * self.data).sum()
                    ans.append((cost, p))
            if self.ansatz_type == 'vertex permutation':
                for v in sampled_params_dicts:
                    p = np.abs(Operator(self.ansatz_qc.bind_parameters(v)).data)
                    p = np.round(p)
                    cost = ((p @ self.data.reshape(int(np.sqrt(self.n)), int(np.sqrt(self.n))) @ p.T).flatten() * self.sum_vec).sum()
                    ans.append((cost, p))
            ans.sort(key=lambda x:x[0], reverse=max)
            answer.append(ans[0])
            # file = open("loss.txt", "a")
            # file.write("{}, {}\n".format(calc(xk), ans[0][0]))
        
        optim = SLSQP(maxiter=maxiter, callback=callback)
        bounds = []
        for i in range(len(qc.parameters)):
            bounds.append((0, np.pi))
        obj = optim.minimize(calc, init_point, bounds=bounds)
        
        optimal_parameters = obj.x

        if self.ansatz_type == 'permutation':
            sampled_params_dicts = sample_exact_thetas(optimal_parameters,
                                                        n=16, seed=np.random.randint(2147483647))
            for v in sampled_params_dicts:
                p = np.abs(Operator(self.ansatz_qc.bind_parameters(v)).data)
                p = np.round(p)
                cost = (self.sum_vec @ p * self.data).sum()
                answer.append((cost, p))
            answer.sort(key=lambda x:x[0], reverse=max)
            return answer[0][1]
        if self.ansatz_type == 'vertex permutation':
            sampled_params_dicts = sample_exact_thetas(optimal_parameters,
                                                        n=16, seed=np.random.randint(2147483647))
            for v in sampled_params_dicts:
                p = np.abs(Operator(self.ansatz_qc.bind_parameters(v)).data)
                p = np.round(p)
                cost = ((p @ self.data.reshape(int(np.sqrt(self.n)), int(np.sqrt(self.n))) @ p.T).flatten() * self.sum_vec).sum()
                answer.append((cost, p))
            answer.sort(key=lambda x:x[0], reverse=max)
            return answer[0][1]
        else:
            return Operator(self.ansatz_qc.bind_parameters(optimal_parameters)).data
