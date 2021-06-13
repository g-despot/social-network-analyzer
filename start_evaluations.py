import os

calculate_embedding = False

io_files = [('git/git.gpickle',
             'git/git_$1.embedding',
             'git/git_$1_$2.csv')]

methods = ['node2vec_snap',
           'node2vec_eliorc',
           'node2vec_custom',
           'deepwalk_phanein',
           'deepwalk_custom']

classifiers = ['logisticalregression',
               'randomforest',
               'gradientboost']

for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)}" +
                      f"--results {io_file[2].replace('$1', method).replace('$2', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding}")
