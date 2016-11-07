import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

passing = passing[passing['season_type'] == 'Regular']

drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position', 'receiving_rec', 'receiving_tar', 'yds_per_rush']

passing.drop(drop_cols, axis=1, inplace=True)

y = passing.pop('DK points')
x = passing.values

x = StandardScaler().fit_transform(x, y)

model = sm.OLS(y, x).fit()


X_train, X_test, y_train, y_test = train_test_split(x, y)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
