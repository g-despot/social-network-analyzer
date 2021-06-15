# Social Netowrk Analyzer

This app implements custom **Node2Vec** and **DeepWalk** algorithms which are evaluated using link prediction, node classification and community detection tasks.

## Prerequisites

You need to install the requirements for this project by running `pip install -r requirements.txt` from the root directory.

To evaluate other graph embedding implementations, you need to place them in the `src/algorithms/` directory.
* [node2vec_snap](https://github.com/aditya-grover/node2vec): Clone it to the `src/algorithms/node2vec` directory.
* [node2vec_deepwalk](https://github.com/phanein/deepwalk): Clone it to the `src/algorithms/deepwalk` directory and follow the [README.rst](https://github.com/phanein/deepwalk#installation) file installation instructions.

Some algorithms don't need to be added via source code, only installed:
* [node2vec_eliorc](https://github.com/eliorc/node2vec): Install it with `pip install node2vec`.

At this time, only these algorithms are supported for evaluation (including the Node2Vec and DeepWalk custom implementations).

## How to run the program

Because of specific Python requirements for multiple packages it's advised to run the program using Docker Compose:

```shell
docker-compose build
docker-compose up
```

To run a specific embedding algorithm with defined arguments:

```shell
python main.py --input git/git.gpickle \
               --output git/git.embedding \
               --results git/link-prediction/git_deepwalk_custom_logisticalregression.csv \
               --method deepwalk_custom \
               --classifier logisticalregression \
               --evaluation link-prediction \
```

To start multiple embedding algorithms in a row with defined arguments edit and run the `start_evaluations.py` script:

```shell
python start_evaluations.py
```

## Application arguments

* `--input`: Input graph in edge list format in the `/input` directory. **Argument required.**
* `--output`: Filename to save the graph embeddings in the `/embeddings` directory. **Argument required.**
* `--results`: Filename to save the evaluation results in the `/results` directory. **Argument required.**
* `--dimensions`: Dimensionality of the word vectors. Default: 128
* `--walk-length`: The number of nodes in each walk. Default: 64
* `--num-walks`: Number of walks from each node. Default: 32
* `--p`: Node2vec return parameter **p**. Default: 2
* `--q`: Node2vec in-out parameter **q**. Default: 1
* `--workers`: Number of worker threads to train the model. Default: 1
* `--seed`: A seed for the random number generator. Default: 0
* `--test-percentage`: Percentage of graph edges that should be used for testing classifiers. Default: 0.1
* `--train-percentage`: Percentage of graph edges that should be used for training classifiers. Default: 0.1
* `--window-size`: Maximum distance between the current and predicted word within a sentence. Default: 10
* `--weighted`: Denotes if the graph is weighted. Default: False
* `--directed`: Denotes if the graph is directed. Default: False
* `--iter`: Number of iterations (epochs) over the corpus. Default: 1
* `--method`: The graph embedding algorithm and specific implementation. Choices: `node2vec_snap`, `node2vec_eliorc`, `node2vec_custom`,  `deepwalk_phanein` and `deepwalk_custom`. **Argument required.**
* `--community-method`: The community detection method for calculating the number of
clusters. Choices: `label-propagation` and `girvan-newman`.
* `--evaluation`: The social network analysis technique to be used.. Choices: `link`,  `randomforest` and `gradientboost`. **Argument required.**
* `--classifier`: The classifier for evaluation. Choices: `logisticalregression`,  `randomforest` and `gradientboost`. **Argument required.**
* `--embed`: Denotes if the embedding should be calculated or loaded from an existing file. Default: False
* `--node-ml-target`: The node target label for classification. Default: 'ml_target'
* `--k`: Number of node samples to estimate betweenness. Default: None
* `--converging`: Iteration when to cut off the Girvan-Newman algorithm if modularity is decreasing. Default: 10