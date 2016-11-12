import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from Final_DF import FinalDF
from ClusterClass import ClusterClass
from sklearn.cluster import AgglomerativeClustering

cc = ClusterClass(position='QB')

df = cc.get_df()
df_14 = df[df['season_year'] == 2014]
df_15 = df[df['season_year'] == 2015]
df_16 = df[df['season_year'] == 2016]

met_cols = ['h/a', 'passing_att', 'passing_yds', 'passing_tds', 'team_score']
p_col = ['points_per_dollar']
sal_p_cols = ['DK salary', 'points_per_dollar']
all_cols = ['h/a', 'passing_att', 'passing_yds', 'passing_tds', 'team_score', 'passing_cmp', 'passing_int', 'passing_twopta', 'passing_twoptm', 'rushing_att', 'rushing_yds', 'rushing_tds', 'rushing_twoptm', 'team_score', 'opp_score', 'total_points', 'DK points', 'DK salary', 'points_per_dollar']

k_14_met = cc.get_cluster(df_14, met_cols)
k_14_points = cc.get_cluster(df_14, p_col)
k_14_sal_points = cc.get_cluster(df_14, sal_p_cols)
k_14_all = cc.get_cluster(df_14, all_cols)

k_14_df = cc.cluster_labels_df(df_14, k_14_met, 'metrics_labels')
k_14_df = cc.cluster_labels_df(df_14, k_14_points, 'points_labels')
k_14_df = cc.cluster_labels_df(df_14, k_14_sal_points, 'sal_points_labels')
k_14_df = cc.cluster_labels_df(df_14, k_14_all, 'all_labels')

sort_14 = k_14_df.sort_values('all_labels', ascending=False)


# hierarchical clustering
x = df_14[all_cols]

ac = AgglomerativeClustering(n_clusters=3)
ac.fit(x)

k_14_df['ac_labels'] = ac.labels_

sort_14_ac = k_14_df.sort_values('ac_labels', ascending=False)
