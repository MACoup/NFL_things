import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pymc3 as pm
import statsmodels.api as sm
from statsmodels.graphics import regressionplots

filepath = '../nfldb_queries/Data/'

passing = pd.read_csv(filepath + 'passing.csv')


passing = passing[passing['season_type'] == 'Regular']

# passing regression with statsmodels
y = passing['DK points']
x = passing['team_score']
x_c = sm.add_constant(x)

model_1 = sm.OLS(y, x_c).fit()

x2 = passing[['team_score', 'passing_att']]
x2_c = sm.add_constant(x2)

model_2 = sm.OLS(y, x2_c).fit()


# plotting
# plt.figure(figsize=(10, 8))
# plt.scatter(x, y, s=30, label='data')
#
# plt.plot(x, model_1.predict(x), label='Ordinary Least Squares Line', lw=3, color='r')
# plt.legend(loc='best')

# more passing regress with score percentage
passing['score_percentage'].fillna(0, inplace=True)
x3 = passing['score_percentage']
x3_c = sm.add_constant(x3)

model_3 = sm.OLS(y, x3_c).fit()

x3a = passing[['passing_att', 'score_percentage']]
x3a_c = sm.add_constant(x3a)

model_3a = sm.OLS(y, x3a_c).fit()


x4 = passing[['team_score', 'passing_att', 'score_percentage']]
x4_c = sm.add_constant(x4)

model_4 = sm.OLS(y, x4_c).fit()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(x3, y, label='data', alpha=0.2, c = 'r')
ax.plot(x3, model_3.predict(x3_c), label='Ordinary Least Squares', lw=3, color='b')
ax.set_ylabel('DK points')
ax.set_xlabel('score_percentage')
plt.legend(loc='best')

# completion percentage to passing_yds
x5 = passing['cmp_percentage']
x5_c = sm.add_constant(x5)
y2 = passing['passing_yds']

model_5 = sm.OLS(y2, x5_c).fit()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(x5, y2, label='data', alpha=0.2, c = 'r')
ax.plot(x5, model_5.predict(x5_c), label='Ordinary Least Squares', lw=3, color='b')
ax.set_ylabel('passing_yds')
ax.set_xlabel('cmp_percentage')
plt.legend(loc='best')

# completion percentage to touchdowns
x6 = passing['cmp_percentage']
x6_c = sm.add_constant(x6)
y3 = passing['passing_tds']

model_6 = sm.OLS(y3, x6_c).fit()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)
# ax.scatter(x_t, y_t, label='data', alpha=0.2, c='r')
# ax.plot(x_t, model_t.predict(x_t), label='Ordinary Least Squares', lw=3, color='b')
# ax.set_ylabel('passing_tds')
# ax.set_xlabel('cmp_percentage')
# plt.legend(loc='best')

# completion percentage to DK points
