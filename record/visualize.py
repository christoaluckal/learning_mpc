import pickle
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True,precision=3)
n = 12
fin_data = []
for i in range(n):
    try:
        with open(f'data_{i}.pkl','rb') as f:
            fin_data.extend(pickle.load(f))
    except:
        print(f"Error loading proc:",i)
    
    
data_array = np.array(fin_data)

import matplotlib.pyplot as plt

vs = data_array[:,3]
plt.hist(vs, bins=10)
plt.xlabel('Linear Velocity')
plt.ylabel('Frequency')
plt.show()