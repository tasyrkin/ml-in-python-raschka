
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,200)
#y = x ** 2
plt.plot(x, x ** 2, label='x ** 2')
plt.plot(x, x ** 1./2, label='x ** 1./2')
plt.plot(x, np.log(x), label='log(x)')
plt.legend(loc='center')
plt.show()
