
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,10,100)
y = np.log(x)
plt.plot(x, y, label = 'logarithm')
plt.legend()
plt.show()
