from EDA.Lineup_Class import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import itertools


'''
This is an exploratory script for the contests that I have personally entered. The goal was to come up with the best lineup based on salary and point production

'''



contest_folder = 'Data/dk_sun_mon_contests/'

point_folder = 'Data/Player_DK_Salaries/'

# Find data sets
contest_week5 = contest_folder + 'contest-standings-sunday-monday-night-week5.csv'

point_week5 = point_folder + 'Week5_Year2016_player_scores2.txt'

contest_week4 = contest_folder + 'contest-standings-sunday-monday-night-week4.csv'

point_week4 = point_folder + 'Week4_Year2016_player_scores2.txt'


# create object
week4 = dfAnalysis(contest_week4, point_week4)
week5 = dfAnalysis(contest_week5, point_week5)

# fit linear models
pp4 = week4.point_percent
x1 = pp4[['DK salary', 'h/a']]
x1 = sm.add_constant(x1)
y1 = pp4['PercentDrafted']

pp5 = week5.point_percent
x = pp5[['DK salary', 'h/a']]
x = sm.add_constant(x)
y = pp5['PercentDrafted']

sal_loc4_model = sm.OLS(y1, x1).fit()
sal_loc4_model_resids = sal_loc4_model.outlier_test()

sal_loc5_model = sm.OLS(y, x).fit()
sal_loc5_model_resids = sal_loc5_model.outlier_test()


week4.knapsack = (48800,
 ['Ben Roethlisberger',
  "Le'Veon Bell",
  'Antonio Brown',
  'Jerick McKinnon',
  'Kyle Rudolph',
  'David Johnson',
  'Sammie Coates',
  'Tyreek Hill',
  'Pittsburgh Defense'])

week5.knapsack = (48800,
 ['Greg Olsen',
  'Randall Cobb',
  'Jacquizz Rodgers',
  'Mike Evans',
  'Cameron Artis-Payne',
  'Davante Adams',
  'Aaron Rodgers',
  'Odell Beckham Jr.',
  'Tampa Bay Defense'])

# Fit ppd vs tvi
x4 = pp4['points_per_dollar']
y4 = pp4['total_value_inversed']
x4 = sm.add_constant(x4)
val_ppd4_model = sm.OLS(y4, x4).fit()

x5 = pp5['points_per_dollar']
y5 = pp5['total_value_inversed']
x5 = sm.add_constant(x5)
val_ppd5_model = sm.OLS(y5, x5).fit()
'''
Week 5 showed much more correlation between point per dollar and total value than did week 4. In fact, when sorting both point percent dataframes on total value inversed, week 5's knapsack lineup had more of the top value players than did week 4.
'''

def plot_value(pp):
    x = pp['points_per_dollar']
    y = pp['total_value_inversed']
    labels = pp['Player']
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    for label, x, y in itertools.izip(labels, x, y):
        ax.annotate(label, xy = (x, y))
    ax.set_xlabel('Point per Dollar')
    ax.set_ylabel('Total_value')
    plt.show()
