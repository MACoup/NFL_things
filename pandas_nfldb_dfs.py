from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn

pd.set_option('display.max_columns', 30)

passing_df = pd.read_csv('Data/passing_df.csv')
receiving_df = pd.read_csv('Data/receiving_df.csv')

# create new column for the player's team's score
def team_score(row):
    if row['team'] == row['home_team']:
        return row['home_score']
    else:
        return row['away_score']

def DK_passing_bonus(row):
    if row['passing_yds'] >= 300:
        row['DK points'] = 3
        return row['DK points']
    else:
        row['DK points'] = 0
        return row['DK points']

def DK_receiving_rush_bonus(row):
    if row['receiving_yds'] >= 100:
        row['DK points'] = 3
        return row['DK points']
    if row['rushing_yds'] >= 100:
        row['DK points'] = 3
        return row['DK points']
    else:
        row['DK points'] = 0
        return row['DK points']

# get team score for each player
passing_df['team_score'] = passing_df.apply(lambda row: team_score(row), axis=1)

receiving_df['team_score'] = receiving_df.apply(lambda row: team_score(row), axis=1)

# get touchdown points for each player
passing_df['total_td_points'] = passing_df['passing_tds'] * 6
receiving_df['total_td_points'] = receiving_df['receiving_tds'] * 6

# get score percentage
passing_df['td_score_percentage'] = passing_df['total_td_points']/passing_df['team_score']
receiving_df['total_score_percentage'] = receiving_df['total_td_points']/passing_df['team_score']


drop_cols = ['home_team', 'home_score', 'away_team', 'away_score']

passing_df.drop(drop_cols, axis=1, inplace=True)
receiving_df.drop(drop_cols, axis=1, inplace=True)


def plot_score_percentage():
    x = passing_df_2016_total['td_score_percentage']
    y = passing_df_2016_total['DK points']
    labels = passing_df_2016_total.index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
    ax.set_xlabel('TD score percentage')
    ax.set_ylabel('DK points')
    plt.show()

def plot_total_td_points():
    x = passing_df_2016_total['total_td_points']
    y = passing_df_2016_total['DK points']
    labels = passing_df_2016_total.index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
    ax.set_xlabel('TD score weighted value')
    ax.set_ylabel('DK points')
    plt.show()

def plot_td():
    x = passing_df_2016_total['passing_tds']
    y = passing_df_2016_total['DK points']
    labels = passing_df_2016_total.index
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
    ax.set_xlabel('TDs')
    ax.set_ylabel('DK points')
    plt.show()

def DK_points(df):
    points = (df['passing_yds']/25) + (df['passing_tds']*4) + (df['passing_twoptm'] * 2) + (df['rushing_yds']/10) + (df['rushing_tds'] * 6) + (df['receiving_yds']/10) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['rushing_yds']/10) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['punt_ret_tds'] * 6 )- (df['passing_int'] * 2) - (df['fumbles_total']) - (df['rushing_loss_yds']/10)
    return points

def passing_data(df, year=None, week=None, player=None):
    if year:
        df = df[df['season_year'] == year]
    else:
        pass
    if week:
        df = df[df['week'] == week]
    else:
        pass
    if player:
        df = df[df['full_name'] == player]
    else:
        pass
    df['DK points'] = df.apply(lambda row: DK_passing_bonus(row), axis=1)
    df['tds_f_pts'] = df['passing_tds'] * 4
    df['yds_f_pts'] = df['passing_yds'] * 0.04
    df['DK points'] = df['DK points'] + (df['passing_yds']* 0.04) + (df['passing_tds']*4) + (df['passing_twoptm'] * 2) + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['punt_ret_tds'] * 6 )- (df['passing_int']) - (df['fumbles_total'])
    return df

def rec_rush_data(df, year=None, week=None, player=None):
    if year:
        df = df[df['season_year'] == year]
    else:
        pass
    if week:
        df = df[df['week'] == week]
    else:
        pass
    if player:
        df = df[df['full_name'] == player]
    else:
        pass
    df['DK points'] = df.apply(lambda row: DK_receiving_rush_bonus(row), axis=1)
    df['DK points'] = df['DK points'] + df['receiving_rec'] + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['punt_ret_tds'] * 6 ) - (df['fumbles_total'])
    return df

if __name__ == '__main__':
    week6_rec_rush = rec_rush_data(receiving_df, year=2016, week=6)
    week6_pass = passing_data(passing_df, year=2016, week=6)
