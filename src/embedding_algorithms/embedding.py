import argparse
import networkx as nx
import pathlib
from gensim.models import Word2Vec
from node2vec import Node2Vec
from src.embedding_algorithms.custom.deepwalk import DeepWalk as CustomDeepWalk
from src.embedding_algorithms.custom.node2vec import Node2Vec as CustomNode2Vec
from src.embedding_algorithms.node2vec.src import node2vec


def create_embedding(args: argparse.Namespace, G: nx.classes.graph.Graph) -> None:
    """Create embeddings from a NetworkX graph and save them.

    Args:
        args (argparse.Namespace): The provided application arguments.
        G (networkx.classes.graph.Graph): The NetworkX graph object.
    """

    if args.method == 'node2vec_snap':
        """Implementation: https://github.com/aditya-grover/node2vec (SNAP - Stanford Network Analysis Project)."""

        G = node2vec.Graph(G, args.directed, args.p, args.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks,
                         size=args.dimensions,
                         window=args.window_size,
                         seed=args.seed,
                         workers=args.workers,
                         iter=args.iter)
        model.wv.save_word2vec_format(args.output)

    elif args.method == 'node2vec_eliorc':
        """Implementation: https://github.com/eliorc/node2vec."""

        node2vecTmp = Node2Vec(graph=G,
                               walk_length=args.walk_length,
                               num_walks=args.num_walks,
                               dimensions=args.dimensions,
                               workers=args.workers)
        model = node2vecTmp.fit(window=args.window_size,
                                seed=args.seed,
                                workers=args.workers,
                                iter=args.iter)
        model.wv.save_word2vec_format(args.output)

    elif args.method == 'node2vec_custom':
        """Custom implementation of the node2vec algorithm."""

        model = CustomNode2Vec(num_walks=args.num_walks,
                               walk_length=args.walk_length,
                               p=args.p,
                               q=args.q,
                               size=args.dimensions,
                               window=args.window_size,
                               seed=args.seed,
                               workers=args.workers,
                               iter=args.iter)
        model.fit(G=G)
        model.save_embedding(args.output)

    elif args.method == 'deepwalk_phanein':
        """Implementation: https://github.com/phanein/deepwalk."""

        from src.embedding_algorithms.deepwalk.deepwalk.__main__ import main
        pathi = pathlib.Path(__file__).parent.parent.parent.absolute()

        arguments = ['--format', 'gpickle', '--input', str(pathi / args.input),
                     '--number-walks', str(args.num_walks),
                     '--representation-size', str(
                         args.dimensions), '--walk-length', str(args.walk_length),
                     '--window-size', str(
                         args.window_size), '--workers', str(args.workers),
                     '--seed', str(args.seed), '--output', str(pathi / args.output)]
        main(arguments)

    elif args.method == 'deepwalk_custom':
        """Custom implementation of the deepwalk algorithm."""

        model = CustomDeepWalk(num_walks=args.num_walks,
                               walk_length=args.walk_length,
                               size=args.dimensions,
                               window=args.window_size,
                               seed=args.seed,
                               workers=args.workers,
                               iter=args.iter)
        model.fit(G=G)
        model.save_embedding(args.output)

    else:
        raise ValueError(f'Invalid embedding algorithm: {args.method}')
