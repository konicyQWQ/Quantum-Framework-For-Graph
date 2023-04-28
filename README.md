# Quantum Framework For Combinatorial Optimization Problem Over Graph

A quantum framework for Combinatorial Optimization Problem with linear objective functions. It also can solve graph problem if it's optimization objective function can be represented as linear summation of edge weight.

- a novel method that can encode $n$ nonnegative integer with $logn + loglogw + 1$ qubits. ( $w$ is maximum number in array )
- using a vector $o$ for transforming the linear optimization objective function to quantum circuit.
- using the permutation ansatz (proposed by [Nicola Mariella et.al.](https://arxiv.org/abs/2111.09732)) to find the minimum (or maximum) answer.

## APIs

- `f = Framework()`: get a Framework instance.
- `f.encode(array)`: encode array into quantum circuit.
- `f.sum(vector)`: a vector contains coefficients of the linear optimization objective function. The framework will use it as optimization objective.
- `f.ansatz('permutation' | 'vertex permutation' | ansatz quantum gate)`: 'permutation' for a normal array, 'vertex permutation' for a graph's adjacency matrix, ansatz quantum gate for custom.
- `f.run()`: find the minimum (or maximum) answer.

## Graph Example

A graph can be represented using adjacency matrix and it can be flattend as a one-dimension array. So that the framework can encode it.

Please see `tsp.py` for traveling salesman problem, `lpp.py` for longest path problem, `mis.py` for maximum independent set problem.

## Easy Example

If we want to find the minimum number in array `arr = [4, 3, 9, 8, 7, 4, 5, 1]`.

Here are many permutations of $arr$ array.

- $arr_0 = [3, 4, 9, 8, 7, 4, 5, 1]$
- $arr_1 = [1, 8, 9, 4, 7, 4, 5, 3]$
- ...

The problem can be represented as finding the minimum value of $1 * arr_i[0]$.

Therefore, `Framework.encode(arr)` and `Framework.sum([1, 0, 0, 0, 0, 0, 0, 0])` and `Framework.ansatz('permutation')`.

```python
from src.framework import Framework
import numpy as np

n = 8
arr = np.array([4, 3, 9, 8, 7, 4, 5, 1])

o = [1, 0, 0, 0, 0, 0, 0, 0]

f = Framework()
f.encode(arr)
f.sum(o)
f.set_ansatz("permutation")
p = f.run() # p is the permutation matrix corresponding to answer

index = (p @ np.arange(n)).astype(int)[0]
ans = arr[index]

print("minimum number: ", ans, " index: ", index)
```