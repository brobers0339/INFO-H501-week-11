#Imports
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from time import time

#Global variables
diamonds = sns.load_dataset('diamonds')
numer_cols = diamonds.select_dtypes(include=['number']).columns


def kmeans(X, k):
    '''
    Performs k-means clustering on given data, X, with k number of clusters.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to cluster.
    k : int
        Given number of clusters to form.
        
    Returns
    -------
    centroids : array, shape (k, n_features)
        Coordinates of cluster centers.
    labels : array, shape (n_samples,)
        Labels of each point indicating the cluster to which it belongs.
    '''
    km = KMeans(n_clusters=k)
    km.fit(X)
    
    centroids = km.cluster_centers_
    labels = km.labels_
    
    return (centroids, labels)

def kmeans_diamonds(n, k):
    '''
    Performs k-means clustering on first given n rows of numerical columns.
    
    Parameters
    ----------
    n : int
        Number of first rows to use from the diamonds seaborn dataset.
    k : int
        Given number of clusters to form.
        
    Returns
    -------
    centroids : array, shape -> (k, n_features)
        Coordinates of cluster centers.
    labels : array, shape -> (n_samples,)
        Labels of each point indicating the cluster to which it belongs.
    '''
    first_n_cols = diamonds[numer_cols].iloc[:n]
    
    return kmeans(np.asarray(first_n_cols), k)


def kmeans_timer(n, k, n_iter=5):
    '''
    Performs k-means clustering on first n rows of numerical columns
    of the diamonds dataset for n_iter iterations to measure run time. 
    Then, calculates and returns the average run time.
    
    Parameters
    ----------
    n : int
        Number of first rows to use from the diamonds seaborn dataset.
    k : int
        Given number of clusters to form.
    n_iter : int, optional
        Number of iterations to run the k-means clustering (default is 5).
        
    Returns
    -------
    avg_time : float
        Average run time over the given n_iter iterations.
    '''
    run_times = []
    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n=n, k=k)
        end = time()
        run_times.append(end - start)
    return sum(run_times) / n_iter