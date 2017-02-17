from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os



def load_dfs():
    passing_df = pd.read_csv('Data/NFLDB_queries/passing_df.csv')
    receiving_df = pd.read_csv('Data/NFLDB_queries/receiving_df.csv')
    rushing_df = pd.read_csv('Data/NFLDB_queries/rushing_df.csv')
    tight_end_df = pd.read_csv('Data/NFLDB_queries/tight_end_df.csv')
    defense_df = pd.read_csv('Data/NFLDB_queries/defense_df.csv')
    return passing_df, receiving_df, rushing_df, tight_end_df, defense_df
# create new column for the player's team's score
def team_score(row):
    if row['team'] == row['home_team']:
        return row['home_score']
    else:
        return row['away_score']

def opp_team_score(row):
    if row['team'] == row['home_team']:
        return row['away_score']
    else:
        return row['home_score']

def opp_team(row):
    if row['team'] == row['home_team']:
        return row['away_team']
    else:
        return row['home_team']

def home_away(row):
    if row['team'] == row['home_team']:
        return 1
    else:
        return 0

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

def team_score_player(df):
    df['team_score'] = df.apply(lambda row: team_score(row), axis=1)
    df['opp_score'] = df.apply(lambda row: opp_team_score(row), axis=1)
    df['opp_team'] = df.apply(lambda row: opp_team(row), axis=1)
    df['h/a'] = df.apply(lambda row: home_away(row), axis=1)
    return df


def DK_def_pa_points(row):
    if row['opp_score'] == 0:
        row['DK points'] = 10
    elif row['opp_score'] < 7:
        row['DK points'] = 7
    elif row['opp_score'] < 14:
        row['DK points'] = 4
    elif row['opp_score'] < 21:
        row['DK points'] = 1
    elif row['opp_score'] < 28:
        row['DK points'] = 0
    elif row['opp_score'] < 35:
        row['DK points'] = -1
    else:
        row['DK points'] = -4
    return row['DK points']


def drop_columns(df, drop_cols):
    df.drop(drop_cols, axis=1, inplace=True)
    return df


def passing_data(df):

    df = team_score_player(df)

    df['total_points'] = (df['passing_tds'] * 6) + (df['rushing_tds'] * 6) + (df['passing_twoptm'] * 2) + (df['receiving_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6)

    df['cmp_percentage'] = df['passing_cmp']/df['passing_att']

    df['score_percentage'] = df['total_points']/df['team_score']

    df['yds_per_rush'] = df['rushing_yds'] / df['rushing_att']

    df['DK points'] = df.apply(lambda row: DK_passing_bonus(row), axis=1)

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df['DK points'] + (df['passing_yds']* 0.04) + (df['passing_tds']*4) + (df['passing_twoptm'] * 2) + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['receiving_rec']) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6) - (df['passing_int']) - (df['fumbles_total'])

    return df

def rec_rush_data(df):
    df = team_score_player(df)

    df['total_points'] = (df['receiving_tds'] * 6) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6)

    df['rec_percentage'] = df['receiving_rec']/df['receiving_tar']

    df['score_percentage'] = df['total_points']/df['team_score']

    df['yds_per_rec'] = df['receiving_yds'] / df['receiving_rec']

    df['yds_per_rush'] = df['rushing_yds'] / df['rushing_att']

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df.apply(lambda row: DK_receiving_rush_bonus(row), axis=1)
    df['DK points'] = df['DK points'] + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['receiving_rec']) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6) - (df['fumbles_total'])
    return df

def defense_data(df):
    df = team_score_player(df)

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df.apply(lambda row: DK_def_pa_points(row), axis=1)

    df['DK points'] = df['DK points'] + df['sack'] + (df['ints'] * 2) + (df['fumble_rec'] * 2) + (df['fumble_rec_tds'] * 6) + (df['kickret_tds'] * 6) + (df['puntret_tds'] * 6) + (df['misc_tds'] * 6)  + (df['int_tds'] * 6) + (df['safety'] * 2) + (df['punt_block'] * 2) + (df['fg_block'] * 2) + (df['xp_block'] * 2)
    return df




if __name__ == '__main__':

    pd.set_option('display.max_columns', 30)

    drop_cols = ['home_team', 'home_score', 'away_team', 'away_score']

    passing_df, receiving_df, rushing_df, tight_end_df, defense_df = load_dfs()

    passing = passing_data(passing_df)
    rec = rec_rush_data(receiving_df)
    rush = rec_rush_data(rushing_df)
    te = rec_rush_data(tight_end_df)
    dst = defense_data(defense_df)
    all_stats = passing.append(rec).append(rush).append(te).append(dst)

    passing = drop_columns(passing, drop_cols)
    rec = drop_columns(rec, drop_cols)
    rush = drop_columns(rush, drop_cols)
    te = drop_columns(te, drop_cols)
    dst = drop_columns(dst, drop_cols)
    all_stats = drop_columns(all_stats, drop_cols)

    passing.to_csv('Data/Position_dfs/passing.csv', index=False)
    rec.to_csv('Data/Position_dfs/rec.csv', index=False)
    rush.to_csv('Data/Position_dfs/rush.csv', index=False)
    te.to_csv('Data/Position_dfs/te.csv', index=False)
    dst.to_csv('Data/Position_dfs/dst.csv', index=False)
    all_stats.to_csv('Data/Position_dfs/all_stats.csv', index=False)
