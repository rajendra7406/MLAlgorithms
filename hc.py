#Hierarchical clustering

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Creatind dendograms for choosing optimal no of clusters.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#ward method is to minimize the cluster variances between each cluster
plt.title('dendogram')
plt.xlabel('no of customers')
plt.ylabel('eucledian distances')
plt.show()

#Fitting the dataset to HC
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Visulaizing the clusters
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red', label='cluster1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue', label='cluster2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green', label='cluster3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='magenta', label='cluster4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='cyan', label='cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual income in $')
plt.ylabel('shopping rate')
plt.legend()
plt.show()