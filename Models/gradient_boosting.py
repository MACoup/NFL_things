import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler







# starting with QB
# without scale


def get_feature_matrix(position, drop_cols):

    '''
    INPUT: Desired position matrix, columns to drop.
    OUTPUT: X, Y
    '''

    fin = FinalDF(position=position)
    df_qb = fin.get_df()

    df_qb.drop(drop_cols, axis=1, inplace=True)


    y = df_qb.pop('points_bin')
    x = df_qb

    return x, y


def raw_data_model(x, y):

    '''
    INPUT: x, y
    OUTPUT: Classifier, predicted y
    '''


    X_train, X_test, y_train, y_test = train_test_split(x, y)

    clf = gbc()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, y_pred


def grid_search(classifier, parameters, x, y):

    gs = GridSearchCV(classifier, parameters, cv=5)

    gs.fit(x, y)

    return gs





if __name__ == '__main__':

    drop_cols = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team']

    x, y = get_feature_matrix('QB', drop_cols)

    clf = gbc()

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    parameters = {'max_features': [None, 'auto', 'log2'],  'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}

    gs = grid_search(clf, parameters, X_train, y_train)

    In [145]: gs.best_score_
Out[145]: 0.59257582616568583

In [146]: gs.best_params_
Out[146]: {'learning_rate': 0.1, 'max_features': 'log2'}
