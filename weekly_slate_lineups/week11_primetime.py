import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from Final_DF import FinalDF

teams = ['NE', 'SEA', 'NYG', 'CIN']

df_qb = FinalDF(season_type='Regular', position='QB', year=2016).get_df()
df_qb = df_qb[df_qb['team'].isin(teams)]
grouped_qb = df_qb.groupby('full_name').mean()

df_wr = FinalDF(season_type='Regular', position='WR', year=2016).get_df()
df_wr = df_wr[df_wr['team'].isin(teams)]
grouped_wr = df_wr.groupby('full_name').mean()

df_rb = FinalDF(season_type='Regular', position='RB', year=2016).get_df()
df_rb = df_rb[df_rb['team'].isin(teams)]
grouped_rb = df_rb.groupby('full_name').mean()

df_te = FinalDF(season_type='Regular', position='TE', year=2016).get_df()
df_te = df_te[df_te['team'].isin(teams)]
grouped_te = df_te.groupby('full_name').mean()

# df_dst = FinalDF(season_type='Regular', position='DST', year=2016).get_df()
# df_dst = df_dst[df_dst['team'].isin(teams)]

pats_prop = 27.75
sea_prop = 21.25
cin_prop = 24
nyg_prop = 22
