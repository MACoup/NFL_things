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
passing = passing[passing['season_type'] == 'Regular']
y = passing.pop('DK points')
drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position']
passing.drop(drop_cols, axis=1, inplace=True)
x2 = passing.values
x = sm.add_constant(x2)
# model_1 = sm.OLS(y, x).fit()
drop_cols = ['Unnamed: 0', 'season_year', 'season_type', 'week', 'team', 'full_name', 'position']



# model_2 = sm.OLS(y, x2).fit()


# passing regression with pymc3

with pm.Model() as model:
    '''
    We use Normal distribution for the estimation process because it is symmetrical around 0, giving flexibility to being either positive or negative
    '''
    a = pm.Normal('a', mu=0, sd=20)
    b = pm.Normal('b', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = a*x2 + b

    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)

    # inference
    niter = 1000
    start = pm.find_MAP()
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
    pm.traceplot(trace)

# passing regression with pymc3 GLM formulas
data = dict(x=x2, y=y)

with pm.Model() as model2:
    pm.glm.glm('y ~ x', data)
    step2 = pm.NUTS()
    trace2 = pm.sample(2000, step=step2, progressbar=True)
    pm.traceplot(trace2)

# plotting
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)
# ax.scatter(x2, y, s=30, label='data')
# pm.glm.plot_posterior_predictive(trace2, samples=100,
#                                  label='posterior predictive regression lines',
#                                  c='black', alpha=0.2)
# ax.plot(x, model_1.predict(x), label='Ordinary Least Squares Line', lw=3, color='r')
# ax.set_ylim(0, 5)
# ax.set_xlim(0, 1)
# plt.legend(loc='best')
