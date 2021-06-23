import os

calculate_embedding = True

io_files = [('git/git.gpickle',
             'git/git_$1.embedding',
             'git/$1/git_$2_$3.csv')]

"""
io_files = [('facebook/facebook.gpickle',
             'facebook/facebook_$1.embedding',
             'facebook/$1/facebook_$2_$3.csv')]
"""
"""
methods = ['node2vec_snap',
           'node2vec_eliorc',
           'node2vec_custom',
           'deepwalk_phanein',
           'deepwalk_custom']
"""

methods = ['deepwalk_phanein']

classifiers = ['logisticalregression',
               'randomforest',
               'gradientboost']

evaluation = 'link-prediction'


for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--evaluation {evaluation}")

for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--evaluation {evaluation}")

for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--evaluation {evaluation}")

for io_file in io_files:
    for method in methods:
        for classifier in classifiers:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method)} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--evaluation {evaluation}")