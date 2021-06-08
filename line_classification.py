import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
import pickle


df = pd.read_csv("4_Spectral_Clusters_Cars.csv") 
X = np.array(df[['x','y']])
Y = np.array(df['Spectral_labels'])
model = 'classifier.pkl'
colors = ['purple','red', 'gray', 'blue', 'pink', 'green', 'yellow']

clf = SVC(kernel='rbf')
clf.fit(X, Y)

Y = clf.predict(X)

with open(model, 'wb') as f:
    pickle.dump(clf, f)

plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(X[:, 0], X[:, 1], c = list(Y),cmap=matplotlib.colors.ListedColormap(colors),s=3)
plt.title('Spectral Clustering',fontsize=10)
plt.xlabel('X axis',fontsize=10)
plt.ylabel('Y axis',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
plt.show()