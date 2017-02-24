import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


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
    position = row['position']
    if year == 2016 and team == 'LA' and week == 1:
        team = 'STL'
    if week == 0:
        week = 17
        year -= 1
    inds = np.where((df['season_year'] == year) & (df['opp_team'] == team) & (df['week'] == week) & (df['position'] == position))[0]
    return sum(df.ix[inds,'DK points'])



def get_last_4_weeks(row, df):

    '''
    INPUT: DataFrame row
    OUTPUT: New Dataframe column value

    Gets the fantasy points allowed by the opponent over their last four games.
    '''


    week = row['week'] - 1
    team = row['opp_team']
    year = row['season_year']
    position = row['position']
    new_df = df[df['opp_team'] == team].reset_index()
    try:
        ind = new_df[new_df['week'] == week].index.tolist()[0]
    except IndexError:
        year -= 1
        for i in range(17, 0, -1):
            week = i
            ind = new_df[new_df['week'] == week].index.tolist()[0]
            if ind:
                break
            else:
                continue
    ind_low = ind-3
    ind_high = ind+1
    if ind_low < 0:
        ind_low = 0
    fin_df = new_df.iloc[ind_low:ind_high,]
    return fin_df['DK points'].mean()


def format_fp_allowed(df):

    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Formats the dataframes to get each player's opponent's allowed fantasy points.
    '''


    df['opp_fp_allowed_last_week'] = df.apply(lambda row: get_last_week(row, df), axis=1)
    t0 = time.time()
    df['opp_fp_allowed_last_4_weeks'] = df.apply(lambda row: get_last_4_weeks(row, df), axis=1)
    t1 = time.time()
    t2 = t1-t0
    print t2


def get_mean_column_last_4_weeks(row, df, col):

    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with mean_score_percentage column.

    Creates a new attribute based on the player's average score percentage.
    '''

    player = row['full_name']
    week = row['week'] - 1
    year = row['season_year']
    print player + str(week) + str(year)
    new_df = df[df['full_name'] == player].reset_index(drop=True)
    if week <= 0:
        week = 1
    try:
        ind = new_df[(new_df['week'] == week) & (new_df['season_year'] == year)].index.tolist()[0]
    except IndexError:
        print IndexError
        try:
            year -= 1
            for i in range(17, 0, -1):
                week = i
                print week
                ind = new_df[(new_df['week'] == week) & (new_df['season_year'] == year)].index.tolist()[0]
                if ind:
                    break
                else:
                    continue
        except IndexError:
            week = row['week']
            year = row['season_year']
            ind = new_df[(new_df['week'] == week) & (new_df['season_year'] == year)].index.tolist()[0]
    ind_low = ind-3
    ind_high = ind+1
    if ind_low < 0:
        ind_low = 0
    return new_df.iloc[ind_low:ind_high,][col].mean()



def get_average_everything(df, remove_cols):
    cols = [col for col in df.columns if col not in remove_cols]
    for col in cols:
        df['mean_{}_last_4_weeks'.format(col)] = df.apply(lambda row: get_mean_column_last_4_weeks(row, df, col), axis=1)




if __name__ == '__main__':
    df = pd.read_csv('Data/Position_dfs/passing.csv')
    df = df[df['season_type'] == 'Regular']
    format_fp_allowed(df)
    remove_cols = ['DK points', 'h/a', 'full_name', 'team', 'opp_team', 'position', 'season_type', 'season_year', 'week', 'spread', 'o/u']
