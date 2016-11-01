from Lineup_Class import dfAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm



# Find data sets
contest_week5 = 'Data/contest-standings-sunday-monday-night-week5.csv'

point_week5 = 'Data/Week5_player_scores.txt'


# create object
week5 = dfAnalysis(contest_week5, point_week5)

# fit linear model
pp = week5.point_percent
x = pp[['DK salary', 'h/a']]
x = sm.add_constant(x)
y = pp['PercentDrafted']

sal_loc_model = sm.OLS(y, x).fit()
sal_loc_model_resids = sal_loc_model.outlier_test()
