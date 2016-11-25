import pandas as pd
import numpy as np


lines_2009 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2009.csv')
lines_2010 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2010.csv')
lines_2011 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2011.csv')
lines_2012 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2012.csv')
lines_2013 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2013.csv')
lines_2014 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2014.csv')
lines_2015 = pd.read_csv('Draft_Kings/Data/NFL_lines/nfl_lines_2015.csv')

team_dict_2015 = {'Titans': 'TEN', 'Cardinals': 'ARI', 'Falcons': 'ATL', 'Panthers': 'CAR', 'Bears': 'CHI', 'Cowboys': 'DAL', 'Lions': 'DET', 'Packers': 'GB', 'Rams': 'STL', 'Vikings': 'MIN', 'Saints': 'NO', 'Giants': 'NYG', 'Eagles': 'PHI', '49ers': 'SF', 'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Redskins': 'WAS', 'Chargers': 'SD', 'Steelers': 'PIT', 'Raiders': 'OAK', 'Jets': 'NYJ', 'Patriots': 'NE', 'Dolphins': 'MIA', 'Chiefs': 'KC', 'Jaguars': 'JAC', 'Colts': 'IND', 'Texans': 'HOU', 'Broncos': 'DEN', 'Browns': 'CLE', 'Bills': 'BUF', 'Ravens': 'BAL'}

team_dict_2016 = {'Titans': 'TEN', 'Cardinals': 'ARI', 'Falcons': 'ATL', 'Panthers': 'CAR', 'Bears': 'CHI', 'Cowboys': 'DAL', 'Lions': 'DET', 'Packers': 'GB', 'Rams': 'LA', 'Vikings': 'MIN', 'Saints': 'NO', 'Giants': 'NYG', 'Eagles': 'PHI', '49ers': 'SF', 'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Redskins': 'WAS', 'Chargers': 'SD', 'Steelers': 'PIT', 'Raiders': 'OAK', 'Jets': 'NYJ', 'Patriots': 'NE', 'Dolphins': 'MIA', 'Chiefs': 'KC', 'Jaguars': 'JAC', 'Colts': 'IND', 'Texans': 'HOU', 'Broncos': 'DEN', 'Browns': 'CLE', 'Bills': 'BUF', 'Ravens': 'BAL'}

def to_datetime(df):
    df['Date'] = pd.to_datetime(df['Date'])

def strip(df):
    df.columns = df.columns.str.strip()
    df['Vis Team'] = df['Vis Team'].str.strip()
    df['Home Team'] = df['Home Team'].str.strip()
    return df

def change_vis_team_2015(df):
    df['Vis Team'].replace(to_replace=team_dict_2015, inplace=True)
    df['Home Team'].replace(to_replace=team_dict_2015, inplace=True)

def change_date(df):
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.week
