import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds


def SVD_embedding(G, output_filename, size=100):
    node_list = list(G.nodes())
    adjacency_matrix = nx.adjacency_matrix(G, node_list)
    adjacency_matrix = adjacency_matrix.astype(float)
    # adjacency_matrix = sparse.csc_matrix(adjacency_matrix)
    U, Sigma, VT = svds(adjacency_matrix, k=size)
    Sigma = np.diag(Sigma)
    W = np.matmul(U, np.sqrt(Sigma))
    C = np.matmul(VT.T, np.sqrt(Sigma))
    # print(np.sum(U))
    embeddings = W + C
    vectors = {}
    for id, node in enumerate(node_list):
        vectors[node] = list(np.array(embeddings[id]))

    fout = open(output_filename, 'w')
    node_num = len(vectors.keys())
    fout.write("{} {}\n".format(node_num, size))
    for node, vec in vectors.items():
        fout.write("{} {}\n".format(node,
                                    ' '.join([str(x) for x in vec])))
    fout.close()

    return
