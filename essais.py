import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.pyplot.show()