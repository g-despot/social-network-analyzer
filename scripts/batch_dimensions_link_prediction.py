import os

calculate_embedding = True


io_files = [('facebook/facebook.gpickle',
             'facebook/dimensions/facebook_$1.embedding',
             'facebook/$1/dimensions/facebook_$2_$3.csv')]

methods = ['node2vec_snap']

classifier = 'logisticalregression'

evaluation = 'link-prediction'
dimensions = [10, 20, 30, 40, 50, 60, 70]

for io_file in io_files:
    for method in methods:
        for dimension in dimensions:
            os.system(f"python .\\main.py --input {io_file[0]} " +
                      f"--output {io_file[1].replace('$1', method + '_' + str(dimension))} " +
                      f"--results {io_file[2].replace('$1', evaluation).replace('$2', method).replace('$3', classifier)} " +
                      f"--method {method} --classifier {classifier} --embed {calculate_embedding} " +
                      f"--dimension {dimension} --evaluation {evaluation}")
