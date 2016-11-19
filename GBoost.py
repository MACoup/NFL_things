import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

fin = FinalDF(season_type='Regular', position='QB')
df = fin.get_df()

df.fillna(0, axis=1, inplace=True)

df = df[df['week'] <= 9]

drop_cols = ['Unnamed: 0', 'season_type', 'team', 'full_name', 'position']

df2 = df.drop(drop_cols, axis=1)

collinear_drop_cols = ['passing_tds', 'passing_yds', 'total_points', 'DK points', 'total_points', 'receiving_tar']

df2.drop(collinear_drop_cols, axis=1, inplace=True)

def class_label(row):
    if row['points_per_dollar'] > 3.75:
        return 1
    else:
        return 0
df['class_label'] = df.apply(lambda row: class_label(row), axis=1)
df2['class_label'] = df2.apply(lambda row: class_label(row), axis=1)

df_2014 = df2[df2['season_year'] == 2014]
df_2015 = df2[df2['season_year'] == 2015]
df_2016 = df2[df2['season_year'] == 2016]

df_2016_2 = df[df['season_year'] == 2016]


y_2014 = df_2014['points_per_dollar'].values
x_2014 = df_2014.drop(['class_label', 'points_per_dollar'], axis=1).values

y_2015 = df_2015['points_per_dollar'].values
x_2015 = df_2015.drop(['class_label', 'points_per_dollar'], axis=1).values

y_2014_c = df_2014['class_label']
x_2014_c = df_2014.drop(['class_label', 'points_per_dollar'], axis=1)


y_2015_c = df_2015['class_label']
x_2015_c = df_2015.drop(['class_label', 'points_per_dollar'], axis=1)

y_2016_c = df_2016['class_label']
x_2016_c = df_2016.drop(['class_label', 'points_per_dollar'], axis=1)

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

    print 'CV score: ', gs.best_score_
    return RMSE(gs)

def calc_GBC():
    gbc = GradientBoostingClassifier()

    gbc.fit(x_2014_c, y_2014_c)

    return gbc

def plot_y():
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.hist(y_2015, bins=20)
    plt.show()

gbc = calc_GBC()
df_2016_2['pred_class_label'] = gbc.predict(x_2016_c)
df_16_names = df_2016_2[['full_name', 'DK salary', 'class_label', 'pred_class_label']]
