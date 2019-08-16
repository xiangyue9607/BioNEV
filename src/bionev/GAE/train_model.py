# -*- coding: utf-8 -*-

import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from bionev.GAE.model import GCNModelAE, GCNModelVAE
from bionev.GAE.optimizer import OptimizerAE, OptimizerVAE
from bionev.GAE.preprocessing import construct_feed_dict, preprocess_graph, sparse_to_tuple


# # Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""


class gae_model(object):
    def __init__(self, args):
        super(gae_model, self).__init__()
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.hidden1 = args.hidden
        self.hidden2 = args.dimensions
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.model_selection = args.gae_model_selection
        self.model = None

    def save_embeddings(self, output, node_list):
        self.feed_dict.update({self.placeholders['dropout']: 0})
        emb = self.sess.run(self.model.z_mean, feed_dict=self.feed_dict)
        print(emb.shape)
        fout = open(output, 'w')
        fout.write("{} {}\n".format(emb.shape[0], emb.shape[1]))
        for idx in range(emb.shape[0]):
            fout.write("{} {}\n".format(node_list[idx], ' '.join([str(x) for x in emb[idx, :]])))
        fout.close()

    def train(self, adj):
        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train = adj
        features = sp.identity(adj.shape[0])  # featureless
        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        # Define placeholders
        self.placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Create model
        if self.model_selection == 'gcn_ae':
            self.model = GCNModelAE(self.placeholders, num_features, features_nonzero, self.hidden1, self.hidden2)
        elif self.model_selection == 'gcn_vae':
            self.model = GCNModelVAE(self.placeholders, num_features, num_nodes, features_nonzero, self.hidden1,
                                     self.hidden2)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if self.model_selection == 'gcn_ae':
                opt = OptimizerAE(preds=self.model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm,
                                  learning_rate=self.learning_rate
                                  )
            elif self.model_selection == 'gcn_vae':
                opt = OptimizerVAE(preds=self.model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=self.model,
                                   num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm,
                                   learning_rate=self.learning_rate
                                   )

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(self.epochs):
            t = time.time()
            # Construct feed dictionary
            self.feed_dict = construct_feed_dict(adj_norm, adj_label, features, self.placeholders)
            self.feed_dict.update({self.placeholders['dropout']: self.dropout})
            # Run single weight update
            outs = self.sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=self.feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")
