from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os

pd.set_option('display.max_columns', 30)

passing_df = pd.read_csv('Data/passing_df.csv')
receiving_df = pd.read_csv('Data/receiving_df.csv')
rushing_df = pd.read_csv('Data/rushing_df.csv')
tight_end_df = pd.read_csv('Data/tight_end_df.csv')
defense_df = pd.read_csv('Data/defense_df.csv')

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
passing_df['team_score'] = passing_df.apply(lambda row: team_score(row), axis=1)
passing_df['opp_score'] = passing_df.apply(lambda row: opp_team_score(row), axis=1)
passing_df['opp_team'] = passing_df.apply(lambda row: opp_team(row), axis=1)
passing_df['h/a'] = passing_df.apply(lambda row: home_away(row), axis=1)

receiving_df['team_score'] = receiving_df.apply(lambda row: team_score(row), axis=1)
receiving_df['opp_score'] = receiving_df.apply(lambda row: opp_team_score(row), axis=1)
receiving_df['h/a'] = receiving_df.apply(lambda row: home_away(row), axis=1)

rushing_df['team_score'] = rushing_df.apply(lambda row: team_score(row), axis=1)
rushing_df['opp_score'] = rushing_df.apply(lambda row: opp_team_score(row), axis=1)
rushing_df['h/a'] = rushing_df.apply(lambda row: home_away(row), axis=1)

tight_end_df['team_score'] = tight_end_df.apply(lambda row: team_score(row), axis=1)
tight_end_df['opp_score'] = tight_end_df.apply(lambda row: opp_team_score(row), axis=1)
tight_end_df['h/a'] = tight_end_df.apply(lambda row: home_away(row), axis=1)

defense_df['team_score'] = defense_df.apply(lambda row: team_score(row), axis=1)
defense_df['opp_score'] = defense_df.apply(lambda row: opp_team_score(row), axis=1)
defense_df['h/a'] = defense_df.apply(lambda row: home_away(row), axis=1)


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


drop_cols = ['home_team', 'home_score', 'away_team', 'away_score']

passing_df.drop(drop_cols, axis=1, inplace=True)
receiving_df.drop(drop_cols, axis=1, inplace=True)
rushing_df.drop(drop_cols, axis=1, inplace=True)
tight_end_df.drop(drop_cols, axis=1, inplace=True)
defense_df.drop(drop_cols, axis=1, inplace=True)



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
    df['total_points'] = (df['passing_tds'] * 6) + (df['rushing_tds'] * 6) + (df['passing_twoptm'] * 2) + (df['receiving_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6)

    df['cmp_percentage'] = df['passing_cmp']/df['passing_att']

    df['score_percentage'] = df['total_points']/df['team_score']

    df['yds_per_rush'] = df['rushing_yds'] / df['rushing_att']

    df['DK points'] = df.apply(lambda row: DK_passing_bonus(row), axis=1)

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df['DK points'] + (df['passing_yds']* 0.04) + (df['passing_tds']*4) + (df['passing_twoptm'] * 2) + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['receiving_rec']) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6) - (df['passing_int']) - (df['fumbles_total'])

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
    df['total_points'] = (df['receiving_tds'] * 6) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_twoptm'] * 2) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6)

    df['rec_percentage'] = df['receiving_rec']/df['receiving_tar']

    df['score_percentage'] = df['total_points']/df['team_score']

    df['yds_per_rec'] = df['receiving_yds'] / df['receiving_rec']

    df['yds_per_rush'] = df['rushing_yds'] / df['rushing_att']

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df.apply(lambda row: DK_receiving_rush_bonus(row), axis=1)
    df['DK points'] = df['DK points'] + (df['rushing_yds'] * 0.1) + (df['rushing_tds'] * 6) + (df['rushing_twoptm'] * 2) + (df['receiving_yds'] * 0.1) + (df['receiving_tds'] * 6) + (df['receiving_twoptm'] * 2) + (df['receiving_rec']) + (df['fumble_rec_tds'] * 6) + (df['puntret_tds'] * 6) + (df['kicking_rec_tds'] * 6) + (df['kickret_tds'] * 6) - (df['fumbles_total'])
    return df

def defense_data(df, year=None, week=None, team=None):
    if year:
        df = df[df['season_year'] == year]
    else:
        pass
    if week:
        df = df[df['week'] == week]
    else:
        pass
    if team:
        df = df[df['team'] == team]
    else:
        pass

    pd.get_dummies(df[['season_year', 'season_type', 'week']])

    df['DK points'] = df.apply(lambda row: DK_def_pa_points(row), axis=1)

    df['DK points'] = df['DK points'] + df['sack'] + (df['ints'] * 2) + (df['fumble_rec'] * 2) + (df['fumble_rec_tds'] * 6) + (df['kickret_tds'] * 6) + (df['puntret_tds'] * 6) + (df['misc_tds'] * 6)  + (df['int_tds'] * 6) + (df['safety'] * 2) + (df['punt_block'] * 2) + (df['fg_block'] * 2) + (df['xp_block'] * 2)
    return df


passing = passing_data(passing_df)
rec = rec_rush_data(receiving_df)
rush = rec_rush_data(rushing_df)
te = rec_rush_data(tight_end_df)
dst = defense_data(defense_df)
all_stats = passing.append(rec).append(rush).append(te).append(dst)

if __name__ == '__main__':

    passing.to_csv('Data/passing.csv', index=False)
    rec.to_csv('Data/rec.csv', index=False)
    rush.to_csv('Data/rush.csv', index=False)
    te.to_csv('Data/te.csv', index=False)
    dst.to_csv('Data/dst.csv', index=False)
    all_stats.to_csv('Data/all_stats.csv', index=False)
