import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Final_DF import FinalDF
from ClusterClass import ClusterClass



if __name__ == '__main__':
    cc = ClusterClass(position='QB')
    df = cc.get_df()

    points_cols = ['DK salary', 'points_per_dollar']
    points_k = cc.get_cluster(df, points_cols)

    labels = points_k.labels_

    X = df[['DK salary', 'points_per_dollar']].values

    plots = cc.plot_kmeans_scatter(X, n_clusters=5)
