import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('GTK3Agg')

x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()