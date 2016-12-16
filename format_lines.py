import pandas as pd
import numpy as np
from pandas.tseries.offsets import *


lines_2009 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2009.csv')
lines_2010 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2010.csv')
lines_2011 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2011.csv')
lines_2012 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2012.csv')
lines_2013 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2013.csv')
lines_2014 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2014.csv')
lines_2015 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2015.csv')

team_dict_2015 = {'Bengals': 'CIN', 'Titans': 'TEN', 'Cardinals': 'ARI', 'Falcons': 'ATL', 'Panthers': 'CAR', 'Bears': 'CHI', 'Cowboys': 'DAL', 'Lions': 'DET', 'Packers': 'GB', 'Rams': 'STL', 'Vikings': 'MIN', 'Saints': 'NO', 'Giants': 'NYG', 'Eagles': 'PHI', '49ers': 'SF', 'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Redskins': 'WAS', 'Chargers': 'SD', 'Steelers': 'PIT', 'Raiders': 'OAK', 'Jets': 'NYJ', 'Patriots': 'NE', 'Dolphins': 'MIA', 'Chiefs': 'KC', 'Jaguars': 'JAC', 'Colts': 'IND', 'Texans': 'HOU', 'Broncos': 'DEN', 'Browns': 'CLE', 'Bills': 'BUF', 'Ravens': 'BAL'}

team_dict_2016 = {'Bengals': 'CIN', 'Titans': 'TEN', 'Cardinals': 'ARI', 'Falcons': 'ATL', 'Panthers': 'CAR', 'Bears': 'CHI', 'Cowboys': 'DAL', 'Lions': 'DET', 'Packers': 'GB', 'Rams': 'LA', 'Vikings': 'MIN', 'Saints': 'NO', 'Giants': 'NYG', 'Eagles': 'PHI', '49ers': 'SF', 'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Redskins': 'WAS', 'Chargers': 'SD', 'Steelers': 'PIT', 'Raiders': 'OAK', 'Jets': 'NYJ', 'Patriots': 'NE', 'Dolphins': 'MIA', 'Chiefs': 'KC', 'Jaguars': 'JAC', 'Colts': 'IND', 'Texans': 'HOU', 'Broncos': 'DEN', 'Browns': 'CLE', 'Bills': 'BUF', 'Ravens': 'BAL'}

def to_datetime(df):
    df['Date'] = pd.to_datetime(df['Date'])

def strip(df):
    df.columns = df.columns.str.strip()
    df['Vis Team'] = df['Vis Team'].str.strip()
    df['Home Team'] = df['Home Team'].str.strip()
    return df

def change_team_2015(df):
    df['Vis Team'].replace(to_replace=team_dict_2015, inplace=True)
    df['Home Team'].replace(to_replace=team_dict_2015, inplace=True)

def get_season_yrs_wks(df):
    df['Year'] = get_year(df)
    df['week'] = df['Date'].map(get_weeks(df))
    df.dropna(axis=0, inplace=True)
    df['week'] = df['week'].astype(int)

def get_year(df):
    year = df['Date'][0].year
    return year

def get_weeks(df):
    day1 = df.iloc[0,0]
    week_dict = {}
    for i in range(1, 18):
        for n in range(7):
            week = pd.date_range(day1, day1 + Week())
            week_dict[week[n]] = i
        day1 = week[-1]
    return week_dict

def vis_df(df):
    df2 = df[['Vis Team', 'Vis Spread', 'Over Under', 'Year', 'week']]
    df2.rename(index=str, columns = {'Vis Team': 'team', 'Vis Spread': 'spread', 'Over Under': 'o/u', 'Year': 'season_year'}, inplace=True)
    return df2

def home_df(df):
    df2 = df[['Home Team', 'Home Spread', 'Over Under', 'Year', 'week']]
    df2.rename(index=str, columns = {'Home Team': 'team', 'Home Spread': 'spread', 'Over Under': 'o/u', 'Year': 'season_year'}, inplace=True)
    return df2

def append_dfs(df):
    v_df = vis_df(df)
    h_df = home_df(df)
    df = pd.concat([v_df, h_df], axis=0, ignore_index=True)
    return df

def form(df):
    to_datetime(df)
    strip(df)
    change_team_2015(df)
    get_season_yrs_wks(df)
    df = append_dfs(df)
    return df


lines_2009 = form(lines_2009)
lines_2010 = form(lines_2010)
lines_2011 = form(lines_2011)
lines_2012 = form(lines_2012)
lines_2013 = form(lines_2013)
lines_2014 = form(lines_2014)
lines_2015 = form(lines_2015)






all_lines = lines_2009.append(lines_2010).append(lines_2011).append(lines_2012).append(lines_2012).append(lines_2013).append(lines_2014).append(lines_2015)
