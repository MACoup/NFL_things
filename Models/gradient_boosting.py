import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin





'''
This model is used to predict whether or not the player's points will be in the top 75th percentile of all players.

TARGET: points_category

'''



# starting with QB
# without scale
# need to deal with imbalanced classes

class CustomScaler(BaseEstimator,TransformerMixin):

    '''
    Custom class to scale only the non-categorical columns of my dataframe.
    '''


    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler()
        self.columns = columns

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.ix[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


def get_feature_matrix(df, balance=False, k=0, m=0):

    '''
    INPUT: Desired position matrix, boolean for balancing classes, and keys for SMOTE algorithm.
    OUTPUT: X, Y
    '''


    cat_col = 'h/a'

    drop_cols = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team']

    scale_cols = [col for col in df.columns if col not in drop_cols]

    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')

    d_cols = [col for col in df.columns if col in drop_cols]
    df.drop(drop_cols, axis=1, inplace=True)


    y = df.pop('points_category')
    x = df

    scale_cols.remove('h/a')
    if 'points_category' in scale_cols:
        scale_cols.remove('points_category')


    scaler = CustomScaler(columns=scale_cols)
    scaler.fit(x)
    df = scaler.transform(x)

    x = df


    if balance:
        x, y = balance_classes(x, y, k=k, m=m)
        fin_df = pd.concat([pd.DataFrame(x, columns=df.columns), pd.DataFrame(y, columns=['points_category'])], axis=1)


    else:
        return x, y, scaler



def determine_points_per_dollar_cat(df, row, percentile):

    '''
    INPUT: DataFrame, row
    OUTPUT: Boolean
    '''

    if row['points_per_dollar'] >= percentile:
        return 1
    else:
        return 0



def get_second_feature_matrix(df, classifier, drop_cols, balance=True, k=0, m=0):

    '''
    INPUT: Desired position matrix, estimator, columns to drop, values for k and m.
    OUTPUT: X, Y
    '''

    if df.columns[0] == 'Unnamed: 0':
        df.drop('Unnamed: 0', axis=1, inplace=True)

    percentile = df['points_per_dollar'].describe()['75%']

    df['points_per_dollar_category'] = df.apply(lambda row: determine_points_per_dollar_cat(df, row, percentile), axis=1)

    df.drop(drop_cols, axis=1, inplace=True)

    df['predicted_points_category'] = classifier.predict(x)

    y = df.pop('points_per_dollar_category')
    x = df



    if balance:
        x, y = balance_classes(x, y, k=k, m=m)
        fin_df = pd.concat([pd.DataFrame(x, columns=df.columns), pd.DataFrame(y, columns=['points_category'])], axis=1)


    else:
        return x, y




def balance_classes(x, y, k=0, m=0):

    '''
    INPUT: DataFrame, num_neighbors
    OUTPUT: Resamped DataFrame

    Performs SMOTE on minority class.
    '''


    n_n = NearestNeighbors(n_neighbors=k)

    sm = SMOTE(random_state=42, k_neighbors=n_n, m_neighbors=m)

    return sm.fit_sample(x, y)




def grid_search(classifier, parameters, x, y):

    '''
    Runs a gridsearch cross validation algorithm.
    '''

    gs = GridSearchCV(classifier, param_grid=parameters, cv=5)

    gs.fit(x, y)

    return gs


def custom_grid_sampling(df):

    '''
    Custom grid search to optimize SMOTE parameters.
    '''

    score_dict = {}

    for k in np.arange(2, 10, 1):
        for m in np.arange(2, 20, 1):

            x, y = get_feature_matrix(df, drop_cols, balance=False, k=k, m=m)


            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

            clf = GradientBoostingClassifier()

            cv_score = cross_val_score(clf, X_train, y_train, cv=5)


            score_dict[tuple([k, m])] = cv_score.mean()

    desired_number = max(score_dict.values())
    key = None
    for k, v in score_dict.iteritems():
        if v == desired_number:
            key = k
    return score_dict, key


def build_model(x, y):

    '''
    INPUT: feature space, Target variable
    OUTPUT: Fitted model
    '''


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

    gbc = GradientBoostingClassifier(learning_rate=0.11, max_depth=4, max_features='sqrt', verbose=5, n_estimators=400)

    gbc.fit(X_train, y_train)

    return gbc



def predict(clf, df, week=9):

    '''
    INPUT: Fitted estimator, target dataframe.
    OUTPUT:

    Scales the data and predicts the point category for each player.
    '''

    cat_col = 'h/a'

    drop_cols = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team']

    scale_cols = [col for col in df.columns if col not in drop_cols]

    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')

    d_cols = [col for col in df.columns if col in drop_cols]
    df.drop(drop_cols, axis=1, inplace=True)


    y = df.pop('points_category')
    x = df

    scale_cols.remove('h/a')
    if 'points_category' in scale_cols:
        scale_cols.remove('points_category')

    x = scaler.transform(x)

    return clf.predict(x)





def load_dfs():

    '''
    INPUT: None
    OUTPUT:

    '''

    fin = FinalDF(position='QB', load_salaries=False)
    df_qb = fin.get_df()

    df_qb_2016 = df_qb[(df_qb['season_year'] == 2016) & (df_qb['week'] > 8)]

    drop_ind = df_qb_2016.index.tolist()[0]
    df_qb = df_qb.iloc[:3209,:]

    return df_qb, df_qb_2016



if __name__ == '__main__':


    df_qb, target_df = load_dfs()





    # score_dict, key = custom_grid_sampling()

    key = (7, 8)

    x, y, scaler = get_feature_matrix(df_qb, balance=False, k=key[0], m=key[1])



    gbc = build_model(x, y)

    # prediction = predict(gbc, target_df,week=9)

    #
    # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)
    #
    # parameters = {'verbose': np.arange(1, 11, 1), 'max_depth': np.arange(1, 5, 1), 'learning_rate': np.arange(0.01, 0.12, 0.01)}
    #
    # clf = GradientBoostingClassifier(max_features='log2')
    #
    # # gs = grid_search(clf, parameters, X_train, y_train)
    #
    # # gs.best_params_ : {'learning_rate': 0.11, 'max_depth': 4, 'max_features': 'sqrt', 'verbose': 5}
    #
    # clf_2 = GradientBoostingClassifier(learning_rate=0.11, max_features='sqrt', verbose=5, max_depth=4)
    #
    # params_2 = {'n_estimators': np.arange(50, 250, 10)}
    #
    # # gs_2 = grid_search(clf_2, params_2, X_train, y_train)
    #
    # # gs_2.best_params_ : {'n_estimators': 240}
    #
    # clf_3 = GradientBoostingClassifier(learning_rate=0.08, max_features='log2', verbose=10, max_depth=3, n_estimators=240)
    #
    # clf_3.fit(X_train, y_train)
    #
    # clf_3.score(X_test, y_test)
    #
    # fin_2 = FinalDF(position='QB', load_salaries=True)
    # df_qb_2 = fin_2.get_df()
    #
    # drop_col_2 = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team', 'points_per_dollar', 'points_category']

    # x_2, y_2 = get_second_feature_matrix(df_qb_2, )
