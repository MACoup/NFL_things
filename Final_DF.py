import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst

dk_data_root = '/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data'

class FinalDF(object):

    def __init__(self, season_type=None, position=None, year=None, week=None):
        self.position = position
        self.year = year
        self.week = week
        self.season_type = season_type

    def _load_salaries(self):
        df = pd.DataFrame()
        for y in range(2014, 2017):
            for w in range(1, 18):
                new_df = pd.read_csv(dk_data_root + '/player_scores/Week{}_Year{}_player_scores2.txt'.format(w, y), delimiter=';')
                new_df['Name'] = new_df['Name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))
                new_df['h/a'] = new_df['h/a'].map({'h' : 0, 'a' : 1})
                new_df.rename(index=str, columns={'Name': 'full_name', 'Week': 'week', 'Year': 'season_year', 'Pos': 'position'}, inplace=True)
                new_df = new_df[['week', 'season_year', 'full_name', 'position', 'DK salary']]
                df = df.append(new_df)
        df['week'] = df['week'].astype(int)
        df['season_year'] = df['season_year'].astype(int)
        return df

    def _sal_position(self):
        df = self._load_salaries()
        df = df[df['position'] == self.position]
        return df

    def _get_nfldb_df(self):
        if self.position:
            if self.position == 'QB':
                df = passing
            elif self.position == 'WR':
                df = rec
            elif self.position == 'RB':
                df = rush
            elif self.position == 'TE':
                df = te
            elif self.position == 'DST':
                df = dst
        if self.season_type:
            df = df[df['season_type'] == self.season_type]
        if self.year:
            df = df[df['season_year'] == self.year]
        if self.week:
            df = df[df['week'] == self.week]
        return df

    def _merge_df(self):
        df1 = self._get_nfldb_df()
        df2 = self._sal_position()
        df = df1.merge(df2, on=['week', 'season_year', 'position', 'full_name'])
        return df

    def get_df(self):
        df = self._merge_df()
        df['points_per_dollar'] = (df['DK points'] / df['DK salary']) * 1000
        return df
