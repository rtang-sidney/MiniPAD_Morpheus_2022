import numpy as np
import matplotlib.pyplot as plt


a= np.random.rand(5, 4)
c = np.delete(a,3,axis=1)
print(a, "\n", c)
3