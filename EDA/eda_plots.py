import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as scs
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst

# plot Aaron Rodgers DK points for all regular season games 2009 - 2015

a_rodg = passing[passing['full_name'] == 'Aaron Rodgers']
a_rodg = a_rodg[a_rodg['season_type'] == 'Regular']
a_rodg = a_rodg[a_rodg['season_year'] != 2016]

col = 'DK points'

def plot_hist(df, col):
    data = df[col]
    ax = data.hist(bins=20, normed=True, edgecolor='none', figsize=(10,7))
    ax.set_ylabel('Probability Distribution')
    ax.set_title(col)
    return ax

def get_sample_mean_var(df, col):
    data = df[col]
    return data.mean(), data.var(ddof=1)

def plot_mom(df, col, ax=None, gamma=True, normal=True):
    if ax is None:
        ax = plot_hist(df, col)
    samp_mean, samp_var = get_sample_mean_var(df, col)
    data = df[col]
    x_vals = np.linspace(data.min(), data.max())

    if gamma:
        alpha = samp_mean**2 / samp_var
        beta = samp_mean / samp_var
        gamma_rv = scs.gamma(a=alpha, scale=1/beta)
        gamma_p = gamma_rv.pdf(x_vals)
        ax.plot(x_vals, gamma_p, color='r', label='Gamma MOM', alpha=0.6)

    if normal:
        # scipy's scale parameter is standard dev.
        samp_std = samp_var**0.5
        normal_rv = scs.norm(loc=samp_mean, scale=samp_std)
        normal_p = normal_rv.pdf(x_vals)
        ax.plot(x_vals, normal_p, color='g', label='Normal MOM', alpha=0.6)

    ax.set_ylabel('Probability Density')
    ax.legend()
    return ax

def plot_mle(df, col, ax=None, gamma=True, normal=True):
    data = df[col]
    x_vals = np.linspace(data.min(), data.max())

    if ax is None:
        ax = plot_hist(df, col)

    if gamma:
        ahat, loc, bhat = scs.gamma.fit(data, floc=0)
        gamma_rv = scs.gamma(a=ahat, loc=loc, scale=bhat)
        gamma_p = gamma_rv.pdf(x_vals)
        ax.plot(x_vals, gamma_p, color='k', alpha=0.7, label='Gamma MLE')

    if normal:
        mean_mle, std_mle = scs.norm.fit(data)
        normal_rv = scs.norm(loc=mean_mle, scale=std_mle)
        normal_p = normal_rv.pdf(x_vals)
        ax.plot(x_vals, normal_p, color='g', label='Normal MLE', alpha=0.6)

    ax.set_ylabel('Probability Density')
    # uniform axes for rainfall data
    ax.legend()

    return ax

cols = ['DK points', 'passing_yds', 'rushing_yds', 'score_percentage']

def plot_many(df, cols, plot_funcs, gamma=True, normal=True):
    cols_srt = sorted(cols)
    axes = df[cols_srt].hist(bins=20, normed=1,
                    grid=0, edgecolor='none',
                    figsize=(12, 8),
                    layout=(2,2))

    for col, ax in zip(cols_srt, axes.flatten()):
        print col, ax
        for func in plot_funcs:
            samp_mean, samp_var = get_sample_mean_var(df, col)
            print samp_mean, samp_var
            func(df, col, ax=ax, gamma=gamma, normal=normal)
    plt.tight_layout()

    return ax

def plot_many_kde(df, cols):
    cols_srt = sorted(cols)
    axes = df[cols_srt].hist(bins=30, normed=1,
                    grid=0, edgecolor='none',
                    figsize=(12, 8),
                    layout=(2,2))
    for col, ax in zip(cols_srt, axes.flatten()):
        data = df[col]
        density = scs.kde.gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 100)
        kde_vals = density(x_vals)
        ax.plot(x_vals, kde_vals, 'r-')

    return ax

if __name__ == '__main__':
    # plot_many(a_rodg, cols, [plot_mom], normal=True)
    # plot_mle(a_rodg, 'passing_yds')
    # plot_kde(a_rodg, 'DK points')
    plot_many_kde(a_rodg, cols)
    plt.show()
