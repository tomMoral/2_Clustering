import numpy as np
import csv

def load(filename):
    """Permit to load data
    Parameters
    ----------
    filename : str
    """
    X = []
    with open(filename) as f:
        doc = csv.reader(f, delimiter = ' ')
        for row in doc:
            X.append((float(row[0]), float(row[1])))
    X = np.array(X)

    return X


