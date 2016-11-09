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



passing = passing[passing['season_type'] == 'Regular']

passing = passing[passing['week'] <= 8]

passing = passing.groupby(['season_year', 'week']).mean().reset_index()

passing_2015 = passing[passing['season_year'] < 2016]

passing_2016 = passing[passing['season_year'] == 2016]
passing_2016.drop('Unnamed: 0', axis=1, inplace=True)

passing_2015 = passing_2015.groupby(['week']).mean()
passing_2015.drop('Unnamed: 0', axis=1, inplace=True)

y_2016 = passing_2016.pop('DK points').values
x_2016 = passing_2016.values


x_2016 = StandardScaler().fit_transform(x_2016)

y = passing_2015.pop('DK points').values
x = passing_2015.values
x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y)

def RMSE(model):
    train_predicted = model.predict(x)

    train_error = np.sqrt(mean_squared_error(train_predicted, y_2016))

    print 'Training Error: ', train_error

def calc_ENET():
    # Fit your model using the training set
    enet = ElasticNet()

    parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'max_iter': [1000, 2000, 3000, 4000, 5000]}

    gs = GridSearchCV(enet, parameters, scoring='neg_mean_squared_error', cv=5)

    gs.fit(x, y)

    print 'CV Error: ', gs.best_score_
    return RMSE(gs)
