from src.framework import Framework
import networkx as nx
import numpy as np

seed = 124092
np.random.seed(seed=seed)

def RandomGraph(n):
   g = nx.Graph()
   for i in range(n):
      for j in range(i+1, n):
         if np.random.randint(4) <= 0:
            g.add_edge(i, j)

   a = []
   for i in range(n):
      a.append(i)

   adj = nx.adjacency_matrix(g, a).todense()
   return adj


def solveMIS(adj):
    adj = np.copy(adj)

    n = len(adj)

    for i in range(n):
        adj[i][i] = -1

    adj = np.append(adj, np.zeros(n * n).reshape(n, n), axis=1)
    adj = np.append(adj, np.zeros(n * 2*n).reshape(n, 2*n), axis=0)
    adj += 1
    adj = adj.astype(int)

    n = n << 1

    a = np.zeros(n * n)
    for i in range(n >> 1):
        for j in range(n >> 1):
            a[i * n + j] = 1

    f = Framework()
    f.encode(adj.flatten())
    f.sum(a)
    f.set_ansatz("vertex permutation")
    p = f.run(seed=seed)

    fp = (p @ np.arange(n)).astype(int)

    ans = []
    for i in range(n >> 1):
        if fp[i] < (n >> 1):
            ans.append(fp[i])

    return len(ans), ans

adj = RandomGraph(4)
ans, points = solveMIS(adj)

print(adj)
print(ans, points)