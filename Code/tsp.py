from src.framework import Framework
import networkx as nx
import numpy as np
import math

seed = 1295729
np.random.seed(seed=seed)

def RandomCompleteGraph(n, max_len):
    g = nx.random_geometric_graph(
        n, radius=0.4, seed=seed)
    pos = nx.get_node_attributes(g, "pos")

    pos[0] = (0.5, 0.5)

    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            dist = int(dist * max_len)
            g.add_edge(i, j, weight=dist)

    adj = nx.adjacency_matrix(g).todense()

    return adj


def solveTSP(adj):
    n = len(adj)
    a = np.zeros(n * n)
    for i in range(n-1):
        a[i * n + i + 1] = 1
    a[(n-1) * n] = 1

    f = Framework()
    f.encode(adj.flatten())
    f.sum(a)
    f.set_ansatz("vertex permutation")
    p = f.run(seed=seed)

    fp = (p @ np.arange(n)).astype(int)

    ans = 0
    for i in range(n-1):
        ans = ans + adj[fp[i]][fp[i+1]]
    ans = ans + adj[fp[n-1]][fp[0]]

    return ans, fp

adj = RandomCompleteGraph(8, 20)
ans, path = solveTSP(adj)

print(adj)
print(ans, path)