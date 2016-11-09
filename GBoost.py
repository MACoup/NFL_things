import pandas as pd
import numpy as np
from Final_DF import FinalDF
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

fin = FinalDF(season_type='Regular', position='QB')
df = fin.df()

df.fillna(0, axis=1, inplace=True)

df = df[df['week'] <= 9]

drop_cols = ['Unnamed: 0', 'season_type', 'team', 'full_name', 'position']

df.drop(drop_cols, axis=1, inplace=True)

collinear_drop_cols = ['passing_tds', 'passing_yds', 'total_points', 'DK points', 'total_points', 'receiving_tar']

df.drop(collinear_drop_cols, axis=1, inplace=True)

df_2014 = df[df['season_year'] == 2014]
df_2015 = df[df['season_year'] == 2015]



y_2014 = df_2014.pop('points_per_dollar').values
x_2014 = df_2014.values

y_2015 = df_2015.pop('points_per_dollar').values
x_2015 = df_2015.values

def RMSE(model):
    x = x_2014[:264]
    train_predicted = model.predict(x)

    train_error = np.sqrt(mean_squared_error(train_predicted, y_2015))

    print 'Training Error: ', train_error

def calc_GBR():
    gbr = GradientBoostingRegressor()

    parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}

    gs = GridSearchCV(gbr, parameters, scoring='neg_mean_squared_error', cv=5)

    gs.fit(x_2014, y_2014)

    return gs.best_params_
