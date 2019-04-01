from OpenNE import line, grarep, sdne, lap, node2vec
from OpenNE import gf, hope
import ast
from struc2vec import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
from utils import *
import time
import logging
from GAE.train_model import gae_model
from SVD.model import SVD_embedding


def embedding_training(args, train_graph_filename):
    if args.method == 'struc2vec':
        g = read_for_struc2vec(train_graph_filename)
    elif args.method == 'GAE':
        g = read_for_gae(train_graph_filename)
    elif args.method == 'SVD':
        g = read_for_SVD(train_graph_filename, weighted=args.weighted)
    else:
        g = read_for_OpenNE(train_graph_filename, weighted=args.weighted)

    _embedding_training(args, G_=g)

    return


def _embedding_training(args, G_=None, seed=0):
    if args.method == 'struc2vec':
        logging.basicConfig(filename='./src/struc2vec/struc2vec.log', filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        if (args.OPT3):
            until_layer = args.until_layer
        else:
            until_layer = None

        G = struc2vec.Graph(G_, args.workers, untilLayer=until_layer)

        if (args.OPT1):
            G.preprocess_neighbors_with_bfs_compact()
        else:
            G.preprocess_neighbors_with_bfs()

        if (args.OPT2):
            G.create_vectors()
            G.calc_distances(compactDegree=args.OPT1)
        else:
            G.calc_distances_all_vertices(compactDegree=args.OPT1)

        print('create distances network..')
        G.create_distances_network()
        print('begin random walk...')
        G.preprocess_parameters_random_walk()

        G.simulate_walks(args.number_walks, args.walk_length)
        print('walk finished..\nLearning embeddings...')
        walks = LineSentence('random_walks.txt')
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                         workers=args.workers, seed=seed)
        os.remove("random_walks.txt")
        model.wv.save_word2vec_format(args.output)
    elif args.method == 'GAE':
        # initialize necessary parameters
        model = gae_model(args)
        # input the graph data
        model.train(G_)
        # save embeddings
        model.save_embeddings(args.output)
    elif args.method == 'SVD':
        SVD_embedding(G_, args.output, size=args.dimensions)
    else:
        if args.method == 'Laplician':
            model = lap.LaplacianEigenmaps(G_, rep_size=args.dimensions)

        elif args.method == 'GF':
            model = gf.GraphFactorization(G_, rep_size=args.dimensions,
                                          epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)

        elif args.method == 'HOPE':
            model = hope.HOPE(graph=G_, d=args.dimensions)

        elif args.method == 'GraRep':
            model = grarep.GraRep(graph=G_, Kstep=args.kstep, dim=args.dimensions)

        elif args.method == 'DeepWalk':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, window=args.window_size, dw=True)

        elif args.method == 'node2vec':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)

        elif args.method == 'LINE':
            model = line.LINE(G_, epoch=args.epochs,
                              rep_size=args.dimensions, order=args.order)

        elif args.method == 'SDNE':
            encoder_layer_list = ast.literal_eval(args.encoder_list)
            model = sdne.SDNE(G_, encoder_layer_list=encoder_layer_list,
                              alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                              batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)
        print("Saving embeddings...")
        model.save_embeddings(args.output)

    return
