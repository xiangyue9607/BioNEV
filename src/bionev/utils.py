# -*- coding: utf-8 -*-

import copy
import itertools
import random

import networkx as nx
import numpy as np

import bionev.OpenNE.graph as og
import bionev.struc2vec.graph as sg


def read_for_OpenNE(filename, weighted=False):
    G = og.Graph()
    print("Loading training graph for learning embedding...")
    G.read_edgelist(filename=filename, weighted=weighted)
    print("Graph Loaded...")
    return G


def read_for_struc2vec(filename):
    print("Loading training graph for learning embedding...")
    G = sg.load_edgelist(filename, undirected=True)
    print("Graph Loaded...")
    return G


def read_for_gae(filename, weighted=False):
    print("Loading training graph for learning embedding...")
    edgelist = np.loadtxt(filename, dtype='float')
    if weighted:
        edgelist = [(int(edgelist[idx, 0]), int(edgelist[idx, 1])) for idx in range(edgelist.shape[0]) if
                    edgelist[idx, 2] > 0]
    else:
        edgelist = [(int(edgelist[idx, 0]), int(edgelist[idx, 1])) for idx in range(edgelist.shape[0])]
    G=nx.from_edgelist(edgelist)
    node_list=list(G.nodes)
    adj = nx.adjacency_matrix(G, nodelist=node_list)
    print("Graph Loaded...")
    return (adj,node_list)


def read_for_SVD(filename, weighted=False):
    if weighted:
        G = nx.read_weighted_edgelist(filename)
    else:
        G = nx.read_edgelist(filename)
    return G


def split_train_test_graph(input_edgelist, seed, testing_ratio=0.2, weighted=False):
    
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist)
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)

    G_train.remove_nodes_from(nx.isolates(G_train))
    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    assert node_num1 == node_num2
    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)

    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)
    return G, G_train, testing_pos_edges, train_graph_filename


def generate_neg_edges(original_graph, testing_edges_num, seed):
    L = list(original_graph.nodes())

    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges


def load_embedding(embedding_file_name, node_list=None):
    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        print('Nodes with embedding: %s'%node_num)
        embedding_look_up = {}
        if node_list:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                if (node_id in node_list):
                    emb = [float(x) for x in vec[1:]]
                    emb = emb / np.linalg.norm(emb)
                    emb[np.isnan(emb)] = 0
                    embedding_look_up[node_id] = np.array(emb)

            # if len(node_list) != len(embedding_look_up):
            #     diff_nodes=set(node_list).difference(set(embedding_look_up.keys()))
            #     for node in diff_nodes:
            #         emb = np.random.random((int(emb_size)))
            #         emb = emb / np.linalg.norm(emb)
            #         emb[np.isnan(emb)] = 0
            #         embedding_look_up[node] = np.array(emb)

            assert len(node_list) == len(embedding_look_up)
        else:
            for line in f:
                vec = line.strip().split()
                node_id = vec[0]
                embeddings = vec[1:]
                emb = [float(x) for x in embeddings]
                emb = emb / np.linalg.norm(emb)
                emb[np.isnan(emb)] = 0
                embedding_look_up[node_id] = list(emb)
            assert int(node_num) == len(embedding_look_up)
        f.close()
        return embedding_look_up


def read_node_labels(filename):
    fin = open(filename, 'r')
    node_list = []
    labels = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        node_list.append(vec[0])
        labels.append(vec[1:])
    fin.close()
    print('Nodes with labels: %s'%len(node_list))
    return node_list, labels


def split_train_test_classify(embedding_look_up, X, Y, seed, testing_ratio=0.2):
    state = np.random.get_state()
    training_ratio = 1 - testing_ratio
    training_size = int(training_ratio * len(X))
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X_train = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size)]
    Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
    X_test = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size, len(X))]
    Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    np.random.set_state(state)
    return X_train, Y_train, X_test, Y_test


def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred
