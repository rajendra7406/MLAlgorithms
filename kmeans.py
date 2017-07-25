#K means clustering

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the datasets
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    #implementing init as random, arises random initialization trap
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
plt.plot()

#Applying kmeans to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visulizing the clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='cluster1' )
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='green', label='cluster2' )
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='blue', label='cluster3' )
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='magenta', label='cluster4' )
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='cyan', label='cluster5' )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroid' )
plt.title('K-Means clustering')
plt.xlabel('Annual Income in $')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()