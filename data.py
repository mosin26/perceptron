import numpy as np


def read_file(filepath):
    with open(filepath, 'r') as f:
        n = int(f.readline()[:-1].split()[0])
        x, y = [], []
        for i in range(n):
            line = f.readline()[:-1].split()
            x.append(line[:-1])
            y.append(line[-1])
    return np.array(x, dtype=float), np.array(y, dtype=int)
