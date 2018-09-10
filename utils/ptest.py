import pickle
import numpy as np

q_table = np.zeros((1,1,6,3,2));

with open('objs.pkl', 'wb') as f:
	pickle.dump(q_table, f)


