import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm


class Node2Vec():
    """An implementation of the Node2Vec algorithm.

    Args:
        num_walks (int): Number of walks from each node.
        walk_length (int): The number of nodes in each walk.
        p (float): Return parameter p.
        q (float): In-out parameter q.
        size (int): Dimensionality of the word vectors.
        workers (int): Number of worker threads to train the model.
        window (int): Maximum distance between the current and predicted word within a sentence.
        iter (int): Number of iterations (epochs) over the corpus.
        seed (int): A seed for the random number generator.
    """

    def __init__(self,
                 num_walks: int,
                 walk_length: int,
                 p: float,
                 q: float,
                 size: int,
                 workers: int,
                 window: int,
                 iter: int,
                 seed: int):

        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.seed = seed

    def fit(self, G: nx.classes.graph.Graph) -> Word2Vec:
        """Fit and get the embedding algorithm model.

        Args:
            G (networkx.classes.graph.Graph): A NetworkX graph object.

        Returns:
            (gensim.models.word2vec.Word2Vec): The Word2Vec model.
        """

        self.G = G
        walks = []
        for node in tqdm(G.nodes()):
            for _ in range(self.num_walks):
                walk = []
                walk.append(str(node))
                current_node = node
                previous_node = None
                former_neighbors = []

                for _ in range(self.walk_length):
                    current_neighbors = np.array(
                        list(G.neighbors(current_node)))
                    if np.size(current_neighbors) == 0:
                        break
                    probability = np.array([1/self.q] * len(current_neighbors))
                    probability[current_neighbors == previous_node] = 1/self.p
                    probability[(
                        np.isin(current_neighbors, former_neighbors))] = 1

                    next_node = np.random.choice(
                        current_neighbors, 1, p=probability/sum(probability))[0]
                    walk.append(str(next_node))

                    former_neighbors = current_neighbors
                    previous_node = current_node
                    current_node = next_node
                walks.append(walk)

        self.model = Word2Vec(walks,
                              size=self.size,
                              window=self.window,
                              seed=self.seed,
                              workers=self.workers,
                              iter=self.iter)

        return self.model

    def get_embedding(self) -> list:
        """Get the embedding vectors.

        Returns:
            A list of embedding vectors.
        """

        self.embedding = []
        for n in self.G.nodes():
            self.embedding.append(self.model[str(n)])

        return self.embedding

    def save_embedding(self, output_path: str):
        """Save the embedding vectors.

        Args:
            output_path (str): File path to save the embedding vectors.
        """

        self.model.wv.save_word2vec_format(output_path)
