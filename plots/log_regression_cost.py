
import numpy as np
import matplotlib.pyplot as plt

# the cost function must be a convex function in order for the gradient decent be able to converge
# https://www.internalpointers.com/post/cost-function-logistic-regression
# cost = -sum(y^i*log(h(x^i)+(1-y^i)*log(1-h(x^i))

x = np.linspace(-20,20,200)
y1 = -np.log(1/(1 + np.exp(-x)))
plt.plot(x, y1, label = 'logreg cost function (left)')
y2 = -np.log(1 - 1/(1 + np.exp(-x)))
plt.plot(x, y2, label = 'logreg cost function (right)')
alpha = 1
y3 = -np.log(1 - 1/(1 + np.exp(-alpha*x)))-np.log(1/(1 + np.exp(-alpha*x)))
plt.plot(x, y3, label = 'logreg cost function (both)')
plt.legend()
plt.show()
