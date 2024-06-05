import os
import numpy as np


def get_data(path, step=0.00001):
    result = []
    for filename in sorted(os.listdir(path), key=lambda s: int(s.split('.')[0].split('_')[1])):
        time = round(int(filename.split('.')[0].split('_')[1])*step, 3)
        with open(path + '/' + filename, 'r') as file:
            x = []
            y = []
            z = []
            for line in file:
                results = line.strip().split()
                x.append(round(float(results[0]), 3))
                y.append(round(float(results[1]), 3))
                z.append(round(float(results[2]), 3))
        result.append((time, np.array(x), np.array(y), np.array(z)))
    return result
