# Example commands

These are some of the commands that can be executed:

## Preparing data before analysis

```python
python .\prepare_data.py --format adjacency-list --input-edges 'amazon/amazon.csv' --output 'amazon/amazon.gpickle' --column-one 'id_1' --column-two 'id_2'
python .\prepare_data.py --format adjacency-list --input-edges 'youtube/youtube.csv' --output 'youtube/youtube.gpickle' --column-one 'id_1' --column-two 'id_2'
python .\prepare_data.py --format nodes-edges --input-edges 'git/git_edges.csv' --input-nodes 'git/git_target.csv' --output 'git/git.gpickle' --column-one 'id_1' --column-two 'id_2' --node-ml-target 'ml_target'
python .\prepare_data.py --format nodes-edges --input-edges 'facebook/facebook_edges.csv' --input-nodes 'facebook/facebook_nodes.csv' --output 'facebook/facebook.gpickle' --column-one 'id_1' --column-two 'id_2' --node-ml-target 'ml_target'

python .\prepare_data.py --sample True --input amazon/amazon_original.gpickle --output amazon/amazon.gpickle
python .\prepare_data.py --sample True --input youtube/youtube_original.gpickle --output youtube/youtube.gpickle
python .\prepare_data.py --sample True --input git/git_original.gpickle --output git/git.gpickle
python .\prepare_data.py --sample True --input facebook/facebook_original.gpickle --output facebook/facebook.gpickle
```

## Running the main program

```python
python .\main.py --input git/git.gpickle --output git/git_node2vec_snap.embedding --results git/git_node2vec_snap_logisticalregression.csv --method node2vec_snap --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input git/git.gpickle --output git/git_node2vec_eliorc.embedding --results git/git_node2vec_eliorc_logisticalregression.csv --method node2vec_eliorc --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input git/git.gpickle --output git/git_node2vec_custom.embedding --results git/git_node2vec_custom_logisticalregression.csv --method node2vec_custom --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input git/git.gpickle --output git/git_deepwalk_phanein.embedding --results git/git_deepwalk_phanein_logisticalregression.csv --method deepwalk_phanein --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input git/git.gpickle --output git/git_deepwalk_custom.embedding --results git/git_deepwalk_custom_logisticalregression.csv --method deepwalk_custom --classifier logisticalregression --embed true --node-ml-target ml_target

python .\main.py --input facebook/facebook.gpickle --output facebook/facebook_node2vec_snap.embedding --results facebook/facebook_node2vec_snap_logisticalregression.csv --method node2vec_snap --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input facebook/facebook.gpickle --output facebook/facebook_node2vec_eliorc.embedding --results facebook/facebook_node2vec_eliorc_logisticalregression.csv --method node2vec_eliorc --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input facebook/facebook.gpickle --output facebook/facebook_node2vec_custom.embedding --results facebook/facebook_node2vec_custom_logisticalregression.csv --method node2vec_custom --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input facebook/facebook.gpickle --output facebook/facebook_deepwalk_phanein.embedding --results facebook/facebook_deepwalk_phanein_logisticalregression.csv --method deepwalk_phanein --classifier logisticalregression --embed true --node-ml-target ml_target
python .\main.py --input facebook/facebook.gpickle --output facebook/facebook_deepwalk_custom.embedding --results facebook/facebook_deepwalk_custom_logisticalregression.csv --method deepwalk_custom --classifier logisticalregression --embed true --node-ml-target ml_target

python .\main.py --input amazon/amazon.gpickle --evaluation community-detection --method girvan-newman
python .\main.py --input amazon/amazon.gpickle --evaluation community-detection --method label-propagation
```