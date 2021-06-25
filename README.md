
# Social Network Analyzer

![](https://img.shields.io/github/license/g-despot/social-network-analyzer)

This is a simple network analysis app, created for the purpose of evaluating embedding algorithms, but can also be used to perform standard graph analysis techniques. Currently implemented algorithms and techniques:
* Embedding algorithms:
  * node2vec
    * node2vec_custom
    * [node2vec_snap](https://github.com/aditya-grover/node2vec)
    * [node2vec_eliorc](https://github.com/eliorc/node2vec)
  * DeepWalk
    * deepwalk_custom
    * [deepwalk_phanein](https://github.com/phanein/deepwalk)
* Machine learning techniques:
  * Link prediction
  * Node classification
  * K-means clustering
* Community detection
  * Girvan-Newman
  * Label propagation

## Prerequisites

You can install the Python requirements for this project by running:

```shell
pip install -r requirements.txt
```

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

To start multiple embedding algorithms in a row with defined arguments edit and run the scripts int the `/scripts` directory:

```shell
python /scripts/batch_link_prediction.py
```

You can find more examples on how to start the program in [example_commands.md](./example_commands.md).

## Application arguments

* `--input`: Input graph in `.gpickle` format in the `/input` directory. **Argument required.**
* `--output`: Filename to save the graph embeddings in the `/embeddings` directory. **Argument required.**
* `--results`: Filename to save the evaluation results in the `/results` directory. **Argument required.**
* `--dimensions`: Dimensionality of the word vectors. Default: 128
* `--walk-length`: The number of nodes in each walk. Default: 64
* `--num-walks`: Number of walks from each node. Default: 32
* `--p`: The node2vec return parameter **p**. Default: 2
* `--q`: The node2vec in-out parameter **q**. Default: 1
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
clusters. Choices: `girvan_newman_custom`, `label_propagation_custom`, `girvan_newman_nx` and `label_propagation_nx`.
* `--evaluation`: The social network analysis technique to be used. Choices: `link`,  `randomforest` and `gradientboost`. **Argument required.**
* `--classifier`: The classifier for evaluation. Choices: `logisticalregression`,  `randomforest` and `gradientboost`. **Argument required.**
* `--embed`: Denotes if the embedding should be calculated or loaded from an existing file. Default: False
* `--node-ml-target`: The node target label for classification. Default: 'ml_target'
* `--k`: Number of node samples to estimate betweenness. Default: None
* `--converging`: Iteration when to cut off the Girvan-Newman algorithm if modularity is decreasing. Default: 10
* `--visuals`: Denotes if the application should plot figures. Default: True

## Structure

* **docs/**: Sphinx generated documentation of the application.
* **embeddings/**: Node embeddings saved in CSV format.
* **input/**: Graph network input files in `.gpickle` format.
* **results/**: Results of evaluation techniques if such were performed.
* **scripts/**: Python scripts for batch task executions.
* **src/**: Source code of the application.