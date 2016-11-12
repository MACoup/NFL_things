import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from sklearn.cluster import KMeans
from Final_DF import FinalDF

class ClusterClass(object):

    def __init__(self, season_type='Regular', position=None, year=None, week=None):
        self.position = position
        self.year = year
        self.week = week
        self.season_type = season_type

    def get_df(self):
        fin = FinalDF(self.season_type, self.position, self.year, self.week)
        df = fin.get_df()
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0, axis=1)
        return df

    def get_cols(self):
        df = self.get_df()
        cols = df.columns
        return cols

    def get_cluster(self, df, X):
        x = df[X]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        k = KMeans(n_clusters=3).fit(x)
        return k

    def cluster_labels_df(self,df, k, title):
        labels = k.labels_
        df[title] = labels
        return df


    def plot_kmeans_scatter(df, X, n_clusters=3):
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111)
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        for i in range(n_clusters):
            ds = X[np.where(labels==i)]
            ax.plot(ds[:,0], ds[:,1], 'o')
        plt.show()
