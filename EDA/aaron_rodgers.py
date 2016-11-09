import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing

passing.fillna(0, axis=1, inplace=True)

a_rodg = passing[passing['full_name'] == 'Aaron Rodgers']
a_rodg = a_rodg[a_rodg['season_type'] == 'Regular']
a_rodg = a_rodg[a_rodg['week'] <= 8]

a_rodg = a_rodg.groupby(['season_year', 'week']).mean().reset_index()
a_rodg.drop('Unnamed: 0', axis=1, inplace=True)


a_rodg15 = a_rodg[a_rodg['season_year'] < 2016].groupby('week').mean()
a_rodg15.drop(4, axis=0, inplace=True)
a_rodg16 = a_rodg[a_rodg['season_year'] == 2016].groupby('week').mean()


y_rodg_2016 = a_rodg16['DK points'].values
y_rodg_2015 = a_rodg15['DK points'].values
x_rodg_2015 = a_rodg15.drop('DK points', axis=1).values
x_rodg_2015 = StandardScaler().fit_transform(x_rodg_2015)

passing = passing[passing['season_type'] == 'Regular']

passing = passing[passing['week'] <= 8]


passing = passing.groupby(['season_year', 'week']).mean().reset_index()

passing_2015 = passing[passing['season_year'] < 2016]
passing_2015_ = passing[passing['season_year'] < 2016]

passing_2016 = passing[passing['season_year'] == 2016]
passing_2016.drop('Unnamed: 0', axis=1, inplace=True)

passing_2015 = passing_2015.groupby(['week']).mean()
passing_2015.drop('Unnamed: 0', axis=1, inplace=True)
passing_2015.drop(4, axis=0, inplace=True)

y_2016 = passing_2016.pop('DK points').values
x_2016 = passing_2016.values


x_2016 = StandardScaler().fit_transform(x_2016)

y_2015 = passing_2015['DK points'].values
x_2015 = passing_2015.values
x_2015 = StandardScaler().fit_transform(x_2015)

a_rodg_DK = a_rodg['DK points']
passing_DK = passing_2015_['DK points']

a_rodg_diff = a_rodg_DK - passing_DK

def RMSE(model):
    train_predicted = model.predict(x_rodg_2015)

    train_error = np.sqrt(mean_squared_error(train_predicted, y_rodg_2016))

    print 'Training Error: ', train_error

def calc_ENET():
    # Fit your model using the training set
    enet = ElasticNet()

    parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'max_iter': [1000, 2000, 3000, 4000, 5000]}

    gs = GridSearchCV(enet, parameters, scoring='neg_mean_squared_error', cv=5)

    gs.fit(x_rodg_2015, y_rodg_2015)

    print 'CV Error: ', gs.best_score_
    return RMSE(gs)
