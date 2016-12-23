import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



def class_label(row):
    if row['points_per_dollar'] > 3.75:
        return 1
    else:
        return 0


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

def calc_GBC(x, y):
    gbc = GradientBoostingClassifier()

    gbc.fit(x, y)

    return gbc

def plot_y():
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.hist(y_2015, bins=20)
    plt.show()

def load_df(drop_cols, collinear_drop_cols, season_type=None, position=None, year=None, week=None, load_lines=True):
    fin = FinalDF(season_type=season_type, position=position, year=year, week=week, load_lines=load_lines)
    df = fin.get_df()
    df.drop(drop_cols, axis=1, inplace=True)
    df.drop(collinear_drop_cols, axis=1, inplace=True)
    df['class_label'] = df.apply(lambda row: class_label(row), axis=1)
    df.fillna(0, axis=1, inplace=True)
    return df

def get_x_y(df, x_drop_cols, y_cols):
    y = df[y_col].values
    x = df.drop(x_drop_cols, axis=1).values
    x = StandardScaler().fit_transform(x)
    return x, y

if __name__ == '__main__':

    drop_cols = ['season_type', 'team', 'full_name', 'position', 'opp_team', 'season_year', 'week']
    collinear_drop_cols = ['passing_tds', 'passing_yds', 'passing_twoptm', 'rushing_tds', 'total_points', 'DK points', 'total_points', 'receiving_tar']

    # load dataframes
    df_2014 = load_df(season_type='Regular', position='QB', year=2014, drop_cols=drop_cols, collinear_drop_cols=collinear_drop_cols)
    df_2015 = load_df(season_type='Regular', position='QB', year=2015, drop_cols=drop_cols, collinear_drop_cols=collinear_drop_cols)
    df_2016 = load_df(season_type='Regular', position='QB', year=2016, drop_cols=drop_cols, collinear_drop_cols=collinear_drop_cols, load_lines=False)

    df_2016_2 = df_2016

    y_col = 'points_per_dollar'
    x_drop_cols = ['class_label', 'points_per_dollar']
    y_col_c = 'class_label'




    x_2014, y_2014 = get_x_y(df_2014, x_drop_cols=x_drop_cols, y_cols=y_col)

    x_2015, y_2015 = get_x_y(df_2015, x_drop_cols=x_drop_cols, y_cols=y_col)

    x_2014_c, y_2014_c = get_x_y(df_2014, x_drop_cols=x_drop_cols, y_cols=y_col_c)


    x_2015_c, y_2015_c = get_x_y(df_2015, x_drop_cols=x_drop_cols, y_cols=y_col_c)

    x_2016_c, y_2016_c = get_x_y(df_2016, x_drop_cols=x_drop_cols, y_cols=y_col_c)


    gbc = calc_GBC(x_2014_c, y_2014_c)
    df_2016_2['pred_class_label'] = gbc.predict(x_2016_c)
    df_16_names = df_2016_2[['full_name', 'week', 'DK salary', 'class_label', 'pred_class_label']]
