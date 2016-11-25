import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

passing = passing[passing['season_type'] == 'Regular']
drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position', 'passing_yds', 'passing_tds']
passing.replace([np.inf, -np.inf], np.nan)
passing.drop(drop_cols, axis=1, inplace=True)
passing.fillna(0, inplace=True)
passing.round(4)


y = passing.pop('DK points')
x = passing.values


X_train, X_test, y_train, y_test = train_test_split(x, y)


def calc_linear(model):
    # Fit your model using the training set
    linear = model(normalize=True)
    linear.fit(X_train, y_train)

    # Call predict to get the predicted values for training and test set
    train_predicted = linear.predict(X_train)
    test_predicted = linear.predict(X_test)

    train_error = mean_squared_error(train_predicted, y_train)
    test_error = mean_squared_error(test_predicted, y_test)

    return train_error, test_error

def cv(model):
    # Fit your model using the training set
    mod = model

    scores = cross_val_score(mod, X_train, y_train, cv=5)

    return scores

if __name__ == '__main__':
    models = [LinearRegression, Ridge, Lasso]
    for model in models:
        calc_linear(model)
