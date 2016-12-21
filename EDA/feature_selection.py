import pandas as pd
import numpy as np
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


def feature_selection_func(df, drop_cols=None, model='LinearRegression()', feature_selection='RFE'):

    if drop_cols:
        df = df.drop(drop_cols, axis=1, inplace=False)

    df.replace([np.inf, -np.inf], np.nan).fillna(0, axis=1, inplace=True)



    y = df.pop('DK points')
    x = df

    # x = StandardScaler().fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    # fit RFE

    estimator = model
    if feature_selection == 'RFE':
        selector = RFE(estimator, 10)
        selector.fit(X_train, y_train)

        l_cols = zip(selector.ranking_, df.columns)

    new_l = get_cols(l_cols)

    return new_l


if __name__ == '__main__':
    passing = pd.read_csv('../nfldb_queries/Data/passing.csv')

    drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position', 'receiving_rec', 'receiving_tar', 'yds_per_rush', 'opp_team']
    feature_selection_func(passing, drop_cols=drop_cols)
