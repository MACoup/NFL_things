import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import RFE


def get_cols(l):
    col_lst = []
    for num, col in l:
        if num == 1:
            col_lst.append(col)
    return col_lst


def feature_selection(df, season_type=None, drop_cols=None, model='LinearRegression()', feature_selection='RFE'):
    if season_type:
        passing = passing[passing['season_type'] == 'season_type']
    if drop_cols:
        drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position', 'receiving_rec', 'receiving_tar', 'yds_per_rush']

        passing.drop(drop_cols, axis=1, inplace=True)

    y = passing.pop('DK points')
    x = passing.values

    x = StandardScaler().fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    # fit RFE

    estimator = model
    if feature_selection == 'RFE':
        selector = RFE(estimator, 10)
        selector.fit(X_train, y_train)

        l_cols = zip(selector.ranking_, passing.columns)

     new_l = get_cols(l_cols)

     return passing
