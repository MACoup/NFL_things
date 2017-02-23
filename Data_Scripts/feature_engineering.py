import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''

This script is for potential feature engineering of data after the intital data transformations and formatting.

'''



def get_last_week(row, df):

    '''
    INPUT: DataFrame row
    OUTPUT: New Dataframe column value

    Gets the fantasy points allowed by that opponent to that position last week.
    '''


    year = row['season_year']
    team = row['opp_team']
    week = row['week'] - 1
    if year == 2016 and team == 'LA' and week == 1:
        team = 'STL'
    if week == 0:
        week = 17
        year -= 1
    new_df = df[(df['season_year'] == year) & (df['opp_team'] == team) & (df['week'] == week)]
    return sum(new_df['DK points'])



def get_last_4_weeks(row):

    '''
    INPUT: DataFrame row
    OUTPUT: New Dataframe column value

    Gets the fantasy points allowed by the opponent over their last four games.
    '''


    # week we are trying to get
    week = row['week'] - 1
    year = row['season_year']
    team = row['opp_team']
    new_df = df[(df['season_year'] == year) & (df['opp_team'] == team)].reset_index()
    try:
        ind = new_df[new_df['week'] == week].index.tolist()[0]
    except IndexError:
        week -= 1
        ind = new_df[new_df['week'] == week].index.tolist()[0]
    ind_low = ind-3
    ind_high = ind+1
    # index error might be because of negative ind. Need to investigate
    if ind_low < 0:
        ind_low = 0
    fin_df = new_df.iloc[ind-3:ind+1,]
    return fin_df['DK points'].mean()


def format_fp_allowed(df):

    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Formats the dataframes to get each player's opponent's allowed fantasy points.
    '''


    df['opp_fp_allowed_last_week'] = df.apply(lambda row: get_last_week(row, df), axis=1)
    df['opp_fp_allowed_last_4_weeks'] = df.apply(lambda row: get_last_4_weeks(row), axis=1)
