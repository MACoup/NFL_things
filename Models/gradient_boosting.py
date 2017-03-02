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









# starting with QB
# without scale
# need to deal with imbalanced classes


def get_feature_matrix(position, drop_cols, balance=True, k=0, m=0):

    '''
    INPUT: Desired position matrix, columns to drop.
    OUTPUT: X, Y
    '''

    fin = FinalDF(position=position)
    df_qb = fin.get_df()

    df_qb.drop(drop_cols, axis=1, inplace=True)

    y = df_qb.pop('points_category')
    x = df_qb

    if balance:
        return balance_classes(x, y, k=k, m=m)

    else:
        return x, y







def balance_classes(x, y, k=0, m=0):

    '''
    INPUT: DataFrame, num_neighbors
    OUTPUT: Resamped DataFrame

    Performs SMOTE on minority class.
    '''


    n_n = NearestNeighbors(n_neighbors=k)

    sm = SMOTE(random_state=42, k_neighbors=n_n, m_neightbors=10)

    return sm.fit_sample(x, y)




def grid_search(classifier, parameters, x, y):

    gs = GridSearchCV(classifier, parameters, cv=5)

    gs.fit(x, y)

    return gs


def custom_grid_sampling():

    score_dict = {}

    for k in np.arange(2, 10, 1):
        for m in np.arange(2, 20, 1):

            x, y = get_feature_matrix('QB', drop_cols, balance=False, k=k, m=m)


            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

            clf = GradientBoostingClassifier()

            cv_score = cross_val_score(clf, X_train, y_train, cv=5)


            score_dict[tuple([k, m])] = cv_score.mean()

    desired_number = max(score_dict.values())
    key = None
    for k, v in score_dict.iteritems():
        if v == desired_number:
            key = k
    return key

if __name__ == '__main__':

    drop_cols = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team']

    x, y = get_feature_matrix('QB', drop_cols, balance=False, k=5)


    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

    parameters = {'max_feature': ['log2', 'sqrt', None], 'verbose': np.arange(1, 11, 1), 'max_depth': np.arange(1, 11, 1)}

    # gs = grid_search(clf, parameters, X_train, y_train)

    clf = GradientBoostingClassifier(max_features='log2', learning_rate=0.05, verbose=3, max_depth=2).fit(X_train, y_train)

    score_dict = custom_grid_sampling()
