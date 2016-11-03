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


# passing regression with pymc3

with pm.Model() as model:
    '''
    We use Normal distribution for the estimation process because it is symmetrical around 0, giving flexibility to being either positive or negative
    '''
    a = pm.Normal('a', mu=0, sd=20)
    b = pm.Normal('b', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = a*x + b

    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)

    # inference
    niter = 1000
    start = pm.find_MAP()
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
    pm.traceplot(trace)

# passing regression with pymc3 GLM formulas
    data = dict(x=x, y=y)

with pm.Model() as model2:
    pm.glm.glm('y ~ x', data)
    step2 = pm.NUTS()
    trace2 = pm.sample(2000, step2, progressbar=True)
    # pm.traceplot(trace2)

# plotting
plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=30, label='data')
pm.glm.plot_posterior_predictive(trace2, samples=100,
                                 label='posterior predictive regression lines',
                                 c='black', alpha=0.2)
plt.plot(x, model_1.predict(x), label='Ordinary Least Squares Line', lw=3, color='r')
plt.set_ylabel('DK points')
plt.legend(loc='best')
