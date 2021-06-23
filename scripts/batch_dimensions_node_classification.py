import os

calculate_embedding = True


io_files = [('git/git.gpickle',
             'git/dimensions/git_$1.embedding',
             'git/$1/dimensions/git_$2_$3.csv')]

methods = ['node2vec_eliorc']

classifier = 'logisticalregression'

evaluation = 'node-classification'
dimensions = [10, 20, 30, 40, 50, 60, 70]

for io_file in io_files:
    for method in methods:
        for dimension in dimensions:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method + '_' + str(dimension))} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--dimension {dimension} --evaluation {evaluation}")
