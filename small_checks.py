import numpy as np


a = np.arange(0, 6, 1).reshape(3, 2)
b = np.arange(5, 8, 1).reshape(3, 1)
c = np.hstack((a, b))
print(c)
np.random.shuffle(c)
print(c)
a = c[:, 0:2]
b = c[:, 2]
print(a, b)

