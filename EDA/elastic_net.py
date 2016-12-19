import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

filepath = '../nfldb_queries/Data/'

passing = pd.read_csv(filepath + 'passing.csv')

passing = passing[passing['season_type']  == 'Regular']

passing.fillna(0, axis=1, inplace=True)

drop_cols = ['Unnamed: 0', 'Unnamed' 'season_year', 'season_type', 'week', 'team', 'full_name', 'position', 'receiving_rec', 'receiving_tar', 'yds_per_rush', 'passing_yds', 'passing_tds', 'rushing_yds', 'rushing_tds', 'total_points', 'cmp_percentage']

passing.drop(drop_cols, axis=1, inplace=True)

y = passing.pop('DK points')
x = passing.values

ss = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y)

def get_max_iters(X_train=X_train, y_train=y_train, X_test=X_test):
    for num in range(1000, 6000, 1000):
        enet = ElasticNet(max_iter=num)
        print enet

        scores = cross_val_score(enet, X_train, y_train, cv=5)

        pred = enet.fit(X_train, y_train).predict(X_test)

        print scores
        print enet.coef_


enet = ElasticNet()
parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'max_iter': [1000, 2000, 3000, 4000, 5000]}
gs = GridSearchCV(enet, parameters, scoring='neg_mean_squared_error', cv=5)
gs.fit(X_train, y_train)

def RMSE(model):
    train_predicted = model.predict(X_train)
    test_predicted = model.predict(X_test)

    train_error = np.sqrt(mean_squared_error(train_predicted, y_train))
    test_error = np.sqrt(mean_squared_error(test_predicted, y_test))

    print 'Traing Error: ', train_error
    print 'Test Error: ', test_error

def calc_ENET():
    # Fit your model using the training set
    enet = ElasticNet()

    parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'l1_ratio' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'max_iter': [1000, 2000, 3000, 4000, 5000]}

    gs = GridSearchCV(enet, parameters, scoring='neg_mean_squared_error', cv=5)

    gs.fit(X_train, y_train)

    return RMSE(gs)
