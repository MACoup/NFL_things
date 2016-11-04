import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pymc3 as pm
import statsmodels.api as sm
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst


# passing regression with statsmodels
y = passing['DK points']
x = passing['team_score']

model_1 = sm.OLS(y, x).fit()

x2 = passing[['team_score', 'passing_att']]

model_2 = sm.OLS(y, x2).fit()


# plotting
# plt.figure(figsize=(10, 8))
# plt.scatter(x, y, s=30, label='data')
#
# plt.plot(x, model_1.predict(x), label='Ordinary Least Squares Line', lw=3, color='r')
# plt.legend(loc='best')

# more passing regress with score percentage
x3 = passing['score_percentage'].fillna(0)

model_3 = sm.OLS(y, x3).fit()

passing['score_percentage'].fillna(0, inplace=True)
x4 = passing[['team_score', 'passing_att', 'score_percentage']]

model_4 = sm.OLS(y, x4).fit()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(x3, y, label='data', alpha=0.2, c = 'r')
ax.plot(x3, model_3.predict(x3), label='Ordinary Least Squares', lw=3, color='b')
ax.set_ylabel('DK points')
ax.set_xlabel('score_percentage')
plt.legend(loc='best')

# completion percentage to passing_yds
x_ = passing['cmp_percentage']
y_ = passing['passing_yds']

model_ = sm.OLS(y, x_).fit()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(x_, y, label='data', alpha=0.2, c = 'r')
ax.plot(x_, model_.predict(x_), label='Ordinary Least Squares', lw=3, color='b')
ax.set_ylabel('DK points')
ax.set_xlabel('cmp_percentage')
plt.legend(loc='best')
