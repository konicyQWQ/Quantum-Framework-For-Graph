# permutation ansatz
# [1] Nicola Mariella, Andrea Simonetto, *A Quantum Algorithm for the Sub-Graph Isomorphism Problem*, https://arxiv.org/abs/2111.09732#
# [2] https://github.com/qiskit-community/subgraph-isomorphism

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from typing import Union, Tuple
from qiskit.circuit import ParameterVector

IntOrTuple = Union[int, Tuple]

def _hcph(phi: float, qc: QuantumCircuit, qubits: IntOrTuple):
    ctrl, target = qubits if isinstance(qubits, tuple) else (-1, qubits)
    qc.h(target)
    if ctrl >= 0:
        qc.cp(phi, ctrl, target)
    else:
        qc.p(phi, target)
    qc.h(target)

S4_BLOCK_PARCOUNT = 5

def _s4_block(params: np.ndarray) -> QuantumCircuit:
    """Build the basic block S4 for the permutations Ansatz.

    Args:
        params (np.ndarray): The array of parameters.
    """
    params = np.asarray(params).flatten()
    assert params.shape == (S4_BLOCK_PARCOUNT,)
    qc = QuantumCircuit(QuantumRegister(name="q", size=2))
    _hcph(params[0], qc, 0)

    qc.h(1)
    qc.p(params[1], 1)
    qc.cp(params[2], 0, 1)
    qc.h(1)

    _hcph(params[3], qc, (1, 0))
    _hcph(params[4], qc, (0, 1))
    return qc


def _map_qreg(array, qreg: QuantumRegister) -> np.ndarray:
    assert isinstance(qreg, QuantumRegister)
    array = np.asarray(array)
    fun = np.vectorize(lambda i: qreg[i])
    return fun(array)


def _expand_topology(topology, *, qreg: QuantumRegister) -> np.ndarray:
    if isinstance(topology, str):
        if topology in {"linear", "circular"}:
            # v = np.arange(qreg.size - 1)
            # v = np.stack([v, v + 1]).T
            v = np.array([[qreg.size - 1, 0]])
            for i in range(qreg.size):
                for j in range(qreg.size):
                    if i < j:
                        v = np.concatenate([v, np.array([[i, j]])], axis=0)
            # if topology == "circular" and qreg.size > 2:
            #     v = np.concatenate([v, np.array([[qreg.size - 1, 0]])], axis=0)
            topology = v
        else:
            raise ValueError(f"Unrecognized topology: {topology}")
    topology = np.asarray(topology)
    assert topology.ndim == 2 and topology.shape[1] == 2
    return _map_qreg(topology, qreg=qreg)


def params_tensor(shape, *, name="t") -> np.ndarray:
    """Prepare a tensor of circuit parameters."""
    shape = tuple(np.atleast_1d(shape).flatten())
    v = ParameterVector(name=name, length=np.product(shape))
    v = np.array(v.params)
    return v.reshape(shape)


def s4_ansatz(
    topology, *, qreg: Union[QuantumRegister, int], params=None
) -> Tuple[QuantumCircuit, np.ndarray]:
    """Construct the permutations ansatz based on the S4 block.

    Args:
        topology (str, np.ndarray): The topology for the ansatz, see the function
        ansatz() for more information.
        qreg (QuantumRegister, int): The destination quantum register.
        params (np.ndarray): The array of parameters.
    """
    if isinstance(qreg, int):
        qreg = QuantumRegister(qreg)
    assert isinstance(qreg, QuantumRegister)
    topology = _expand_topology(topology, qreg=qreg)
    if params is None:
        params = params_tensor((len(topology), S4_BLOCK_PARCOUNT))
    params = np.asarray(params)
    assert params.ndim == 2 and params.shape[1] == S4_BLOCK_PARCOUNT
    assert len(params) == len(topology)
    qc = QuantumCircuit(qreg)
    for v, q in zip(params, topology):
        qc.compose(_s4_block(v), qubits=q, inplace=True)

    qc_ = QuantumCircuit(qreg)
    qc_.compose(qc.to_gate(label="PermAnsatz"), inplace=True)
    return qc_, params


def thetas_to_prob(x) -> np.ndarray:
    x = np.asarray(x) / np.pi
    x = np.abs(x)
    r = np.modf(x)
    r = r[0], r[1] % 2
    return np.abs(r[0] - r[1])


def sample_exact_thetas(v, *, n=1, seed=None):
    if isinstance(v, dict):
        dkeys = v.keys()
        v = np.array(list(v.values()))
    v = thetas_to_prob(v)
    rng = np.random.default_rng(seed)
    prob = rng.uniform(size=(n, len(v)))
    v = (prob < v) * np.pi
    if dkeys is not None:
        v = [dict(zip(dkeys, v1)) for v1 in v]
        assert len(v) == n
    return v