import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from sklearn.cluster import KMeans

fin = FinalDF(season_type='Regular', position='QB')
df = fin.get_df()
fin_2014 = FinalDF(season_type='Regular', position='QB', year=2014)
df_2014 = fin_2014.get_df()
fin_2015 = FinalDF(season_type='Regular', position='QB', year=2015)
df_2015 = fin_2015.get_df()

passing = passing

def plot_scatter(df, x, y, year=None, week=None):
    if year and week:
        df = df[(df['season_year'] == year) & (df['week'] == week)]
    if year:
        df = df[df['season_year'] == year]
    if week:
        df = df[df['week'] == week]
    x_v = df[x].values
    y_v = df[y].values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x_v, y_v, c='r', alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

x = df['points_per_dollar']
X = df['points_per_dollar'].values[np.where(np.isfinite(x))].reshape(-1,1)
x_2 = df['DK salary'].values.reshape(-1,1)
y_2 = df['DK points'].values.reshape(-1,1)

val_2014 = df_2014[['DK points', 'DK salary', 'h/a', 'team_score']]
x_2014 = np.array(val_2014)
k_2014 = KMeans(n_clusters=3).fit(x_2014)

val_2015 = df_2015[['DK points', 'DK salary', 'h/a', 'team_score']]
x_2015 = np.array(val_2015)

pred = k_2014.predict(x_2015)

df_2015['pred_cluster'] = pred

df_2015_0 = df_2015[df_2015['pred_cluster'] == 0]
df_2015_1 = df_2015[df_2015['pred_cluster'] == 1]
df_2015_2 = df_2015[df_2015['pred_cluster'] == 2]


def plot_kmeans_cluster_selection(X, y=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lst = []
    ns = range(1, 10)
    for n in ns:
        kmeans = KMeans(n_clusters=n).fit(X, y)
        lst.append(kmeans.inertia_)
    p = zip(ns, lst)
    ax.plot(p)
    ax.set_title(title)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    plt.show()

def plot_kmeans_hist(X):
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    kmeans = KMeans(n_clusters=3).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i in range(3):
        ds = X[np.where(labels==i)]
        ax.hist(ds, bins=20, alpha=0.5)
    ax.set_title('Points per dollar Clusters')
    plt.show()


def plot_kmeans_scatter(X, y=None, n_clusters=3):
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    kmeans = KMeans(n_clusters=n_clusters).fit(X, y)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        dsx = X[np.where(labels==i)]
        dsy = y[np.where(labels==i)]
        ax.plot(dsx, dsy, 'o')
    ax.set_ylabel('DK points')
    ax.set_xlabel('DK salary')
    plt.show()

def plot_ap_hist(X):
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111)
    ap = AffinityPropogation().fit(X)
    labels = ap.labels_
    centroids = cluster_centers_
