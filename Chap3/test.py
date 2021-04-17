import numpy as np

a = np.ones([16,28,28,1])
b = np.zeros([1,28,28,1])
c = np.concatenate([a, b])
print(c.shape)