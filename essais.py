import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


a = [[2,0], [4,6]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))