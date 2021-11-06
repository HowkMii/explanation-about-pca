import numpy as np                         # Linear algebra library
import matplotlib.pyplot as plt            # library for visualization
from sklearn.decomposition import PCA      # PCA library
import pandas as pd                        # Data frame library
import math                                # Library for math functions
import random                              # Library for pseudo random numbers


n = 1  # The amount of the correlation
x = np.random.uniform(1,2,1000) 
y = x.copy() * n 
x = x - np.mean(x) 
y = y - np.mean(y) 

data = pd.DataFrame({'x': x, 'y': y}) 
plt.scatter(data.x, data.y) 
pca = PCA(n_components=2) # 
pcaTr = pca.fit(data)

rotatedData = pcaTr.transform(data)
dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2']) 
# Plot the transformed data in orange
plt.scatter(dataPCA.PC1, dataPCA.PC2)
plt.show()



print('Eigenvectors or principal component: First row must be in the direction of [1, n]')
print(pcaTr.components_)

print()
print('Eigenvalues or explained variance')
print(pcaTr.explained_variance_)
