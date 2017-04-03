import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


'''
This script is for aggregating features for offensive positions.
'''

# Defense will need to have special aggregations

def get_last_week(row, df, defense=False):

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



def get_last_4_weeks(row, df, defense=False):

    '''
    INPUT: DataFrame row
    OUTPUT: New Dataframe column value

    Gets the fantasy points allowed by the opponent over their last four games.
    '''


    if defense == True:
        position = row[row['team']]
    else:
        week = row['week'] - 1
        team = row['opp_team']
        year = row['season_year']
        position = row['position']
        new_df = df[df['opp_team'] == team].reset_index()
        try:
            ind = new_df[new_df['week'] == week].index.tolist()[0]
        except IndexError:
            try:
                year -= 1
                for i in range(17, 0, -1):
                    week = i
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
        fin_df = new_df.iloc[ind_low:ind_high,]
        return fin_df['DK points'].mean()




def format_fp_allowed(df, defense=False):

    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Formats the dataframes to get each player's opponent's allowed fantasy points.
    '''


    df['opp_fp_allowed_last_week'] = df.apply(lambda row: get_last_week(row, df, defense=defense), axis=1)
    df['opp_fp_allowed_last_4_weeks'] = df.apply(lambda row: get_last_4_weeks(row, df,  defense=defense), axis=1)



def get_mean_column_last_4_weeks(row, df, col):

    '''
    INPUT: DataFrame row, DataFrame, column to be aggregated
    OUTPUT: Aggregated Statistic

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



def get_mean_column_last_4_weeks_dst(row, df, col):

    '''
    INPUT: DataFrame row, DataFrame, column to be aggregated
    OUTPUT: Aggregated Statistic

    Creates a new attribute based on the player's average column value percentage for the defense/special teams DataFrame.
    '''


    player = row['team']
    week = row['week'] - 1
    year = row['season_year']
    print player + str(week) + str(year)
    new_df = df[df['team'] == player].reset_index(drop=True)
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



def get_average_everything(df, exclude_cols):

    '''
    INPUT: DataFrame and columns to not be aggregated.
    OUTPUT: None

    Provides average stats for each plaer over the last 4 weeks.
    '''


    cols = [col for col in df.columns if col not in remove_cols]
    for col in cols:
        df['mean_{}_last_4_weeks'.format(col)] = df.apply(lambda row: get_mean_column_last_4_weeks(row, df, col), axis=1)


def get_average_everything_dst(df, exclude_cols):

    '''
    INPUT: DataFrame and columns to not be aggregated.
    OUTPUT: None

    Provides average stats for each plaer over the last 4 weeks.
    '''


    cols = [col for col in df.columns if col not in exclude_cols]
    for col in cols:
        df['mean_{}_last_4_weeks'.format(col)] = df.apply(lambda row: get_mean_column_last_4_weeks_dst(row, df, col), axis=1)


def apply_aggs(dfs, exclude_cols):

    '''
    INPUT: DataFrame, Columns to not be considered for Aggregation.
    OUTPUT: None

    Applies all transformations and aggregations to DataFrames.
    '''


    for df in dfs:
        get_average_everything(df, exclude_cols)
        format_last_weekallowed(df, exclude_cols)
        cut_points(df)


def append_all_stats(dfs):

    '''
    INPUT: List of position DataFrames
    OUTPUT: DataFrame of all position DataFrames
    '''

    new_df = pd.DataFrame()
    for df in dfs:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        new_df = new_df.append(df).reset_index(drop=True)
    new_df.sort_values(['season_year', 'week'], inplace=True)
    return new_df


def determine_points_cat(df, row):

    '''
    INPUT: DataFrame, row
    OUTPUT: Boolean
    '''
    week = row['week']
    year = row['season_year']
    percentile = df[(df['season_year'] == year) & (df['week'] == week)]['DK points'].describe()['75%']
    if row['DK points'] >= percentile:
        return 1
    else:
        return 0


def cut_points(dfs):

    '''
    INPUT: DataFrames
    OUTPUT: DataFrames with discretized DK points
    '''

    for df in dfs:
        df['points_category'] = df.apply(lambda row: determine_points_cat(df, row), axis=1).astype('category')



def eliminate_feats(df, remove_cols):

    '''
    INPUT: DataFrames.
    OUTPUT: DataFrames with unwanted features removed.
    '''


    return df.drop(remove_cols, axis=1, inplace=False)


if __name__ == '__main__':
    passing = pd.read_csv('Data/Position_dfs/passing.csv')
    rec = pd.read_csv('Data/Position_dfs/rec.csv')
    rush = pd.read_csv('Data/Position_dfs/rush.csv')
    te = pd.read_csv('Data/Position_dfs/te.csv')


    passing = passing[passing['season_type'] == 'Regular']
    rec = rec[rec['season_type'] == 'Regular']
    rush = rush[rush['season_type'] == 'Regular']
    te = te[te['season_type'] == 'Regular']

    exclude_cols = ['DK points', 'h/a', 'full_name', 'team', 'opp_team', 'position', 'season_type', 'season_year', 'week', 'spread', 'o/u']


    passing_agg = pd.read_csv('Data/Positions_agg/passing_agg.csv')
    rec_agg = pd.read_csv('Data/Positions_agg/rec_agg.csv')
    rush_agg = pd.read_csv('Data/Positions_agg/rush_agg.csv')
    te_agg = pd.read_csv('Data/Positions_agg/rush_agg.csv')



    # dfs = [passing, rec, rush, te]

    dfs = [passing_agg, rec_agg, rush_agg, te_agg]

    # apply_aggs(dfs, exclude_cols)
    # cut_points(dfs)
    # all_offense_agg = append_all_stats(dfs)


    remove_cols = ['passing_att',
 'passing_cmp',
 'passing_yds',
 'passing_int',
 'passing_tds',
 'passing_twopta',
 'passing_twoptm',
 'receiving_rec',
 'receiving_tar',
 'receiving_tds',
 'receiving_twopta',
 'receiving_twoptm',
 'receiving_yac',
 'receiving_yds',
 'rushing_att',
 'rushing_yds',
 'rushing_tds',
 'rushing_loss_yards',
 'rushing_twoptm',
 'fumbles_total',
 'fumble_rec_tds',
 'puntret_tds',
 'kickret_tds',
 'kicking_rec_tds',
 'team_score',
 'opp_score',
 'total_points',
 'cmp_percentage',
 'score_percentage',
 'yds_per_rush']
    #
    # passing_agg = eliminate_feats(passing, remove_cols)
    # rec_agg = eliminate_feats(rec, remove_cols)
    # rush_agg = eliminate_feats(rush, remove_cols)
    # te_agg = eliminate_feats(te, remove_cols)

    directory = 'Data/Positions_agg/'

    passing_agg.to_csv(directory + 'passing_agg.csv', index=False)
    rec_agg.to_csv(directory + 'rec_agg.csv', index=False)
    rush_agg.to_csv(directory + 'rush_agg.csv', index=False)
    te_agg.to_csv(directory + 'te_agg.csv', index=False)
