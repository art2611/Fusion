import numpy as np

import matplotlib
matplotlib.use('TkAgg')


x = np.arange(0, 5, 0.1)
y = np.sin(x)
matplotlib.pyplot.plot(x, y)
matplotlib.pyplot.show()