import numpy as np

a = np.array(range(27))
a = a.reshape(3,3,3)
b = np.random.randint(0,10,(3,3,3))
c = a*b
print(c)