
import numpy as np
import matplotlib.pyplot as plt

# for linear regression the MSE (mean square error) cost function is used.
# such cost function is the parabola


x = np.linspace(-20,20,200)
y = x**2
plt.plot(x, y, label = 'linreg cost function')
plt.legend()
plt.show()

