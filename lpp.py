from src.framework import Framework
import networkx as nx
import numpy as np

seed = 3458962
np.random.seed(seed=seed)

def RandomDirectedGraph(n):
    g = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i != j:
                g.add_edge(i, j, weight=np.random.randint(20) - 10)

    adj = nx.adjacency_matrix(g).todense()
    return adj

def solveLPP(adj):
    ori_adj = adj
    adj = np.copy(adj)
    n = len(adj)

    biggest_number = np.max(adj)
    adj[adj==0] = -biggest_number * (n-1)
    for i in range(n):
        adj[i, i] = 0
    adj = np.append(adj, np.zeros(n * n).reshape(n, n), axis=1)
    down = np.ones(n * n).reshape(n, n) * -biggest_number * (n-1)
    adj = np.append(adj, np.append(down, np.zeros(n * n).reshape(n, n),axis=1), axis=0)
    
    min_number = np.min(adj)
    adj[:,:] -= min_number
    adj = adj.astype(int)

    n = n << 1
    
    a = np.zeros(n * n)
    for i in range(int(n/2) - 1):
        a[i * n + i + 1] = 1

    f = Framework()
    f.encode(adj.flatten())
    f.sum(a)
    f.set_ansatz("vertex permutation")
    p = f.run(max=True, seed=seed)

    fp = (p @ np.arange(n)).astype(int)

    path = []
    ans = 0
    for i in range(n >> 1):
        path.append(fp[i])
        if fp[i+1] >= (n >> 1):
            break
        ans = ans + ori_adj[fp[i]][fp[i+1]]

    return ans, path

def solvek_lengthLPP(adj, k):
    ori_adj = adj
    adj = np.copy(adj)
    n = len(adj)

    biggest_number = np.max(adj)
    adj[adj==0] = -biggest_number * (n-1)
    for i in range(n):
        adj[i, i] = 0
    
    min_number = np.min(adj)
    adj[:,:] -= min_number
    adj = adj.astype(int)

    a = np.zeros(n * n)
    for i in range(k):
        a[i * n + i + 1] = 1

    f = Framework()
    f.encode(adj.flatten())
    f.sum(a)
    f.set_ansatz("vertex permutation")
    p = f.run(max=True, seed=seed)

    fp = (p @ np.arange(n)).astype(int)

    path = []
    ans = 0
    for i in range(k):
        path.append(fp[i])
        ans = ans + ori_adj[fp[i]][fp[i+1]]

    return ans, path

adj = RandomDirectedGraph(8)
# ans, path = solveLPP(adj)
ans, path = solvek_lengthLPP(adj, 3) # a better performance if finding k-length longest path problem

print(adj)
print(ans, path)