from src.framework import Framework
import numpy as np

seed = 23498
np.random.seed(seed=seed)

def randArray(n, max_len):
    arr = []
    for _ in range(n):
        arr.append(np.random.randint(max_len))
    return np.array(arr)


def solveFindMin(arr):
    n = len(arr)
    a = np.zeros(n)
    a[0] = 1

    f = Framework()
    f.encode(arr)
    f.sum(a)
    f.set_ansatz("permutation")
    p = f.run(seed=seed)

    fp = (p @ np.arange(n)).astype(int)
    ans = arr[fp[0]]

    return ans, fp[0]

arr = randArray(32, 20)
ans, index = solveFindMin(arr)

print(arr)
print(ans, index)