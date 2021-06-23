import os

calculate_embedding = True

io_files = [('amazon/amazon.gpickle',
             'amazon/amazon_$1.embedding',
             'amazon/$1/amazon_$2_$3.csv')]

"""
methods = ['node2vec_snap',
           'node2vec_eliorc',
           'node2vec_custom',
           'deepwalk_phanein',
           'deepwalk_custom']
"""
methods = ['node2vec_snap',
           'deepwalk_custom']

dimensions = [3, 6, 9, 12, 15, 18, 21, 24, 27]
community_method = 'label_propagation_nx'
evaluation = 'community-detection'

for io_file in io_files:
    for method in methods:
        for dimension in dimensions:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', community_method)} " +
                      f"--method {method} --community-method {community_method} --embed {calculate_embedding} " +
                      f"--dimension {dimension} --evaluation {evaluation} --visuals False")
