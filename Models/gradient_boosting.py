import pandas as pd
import numpy as np
from Final_DF import FinalDF
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler




drop_cols = ['season_year', 'season_type', 'week', 'full_name', 'position', 'DK points', 'team', 'opp_team']


# starting with QB

fin = FinalDF(position='QB')
df_qb = fin.get_df()

df_qb.drop(drop_cols, axis=1, inplace=True)

cats_calls = ['h/a']

y = df_qb.pop('points_bin')
x = df_qb

X_train, X_test, y_train, y_test = train_test_split(x, y)

clf = gbc()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
