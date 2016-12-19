import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys
sys.path.append(os.path.relpath('nfldb_queries/'))
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from format_lines import lines_2009, lines_2010, lines_2011, lines_2012, lines_2013, lines_2014, lines_2015, all_lines

dk_data_root = 'Draft_Kings/Data'

class FinalDF(object):

    '''
    This creates the DataFrame that will be used in analysis. It combines the position dataframes with the salary dataframes.
    '''

    def __init__(self, season_type=None, position=None, year=None, week=None):
        self.position = position
        self.year = year
        self.week = week
        self.season_type = season_type

    def _load_salaries(self):
        '''
        Formats salary dataframes. Creates a dataframe from each individual salary file, then appends them together.
        '''
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
        df['position'].replace(to_replace='Def', value='DST', inplace=True)
        df['full_name'].replace(to_replace='Odell Beckham Jr.', value='Odell Beckham', inplace=True)
        return df

    def _sal_position(self):
        '''
        Gets the correct position salary dataframe.
        '''
        df = self._load_salaries()
        if self.position:
            df = df[df['position'] == self.position]
        else:
            df = df
        return df

    def _load_lines(self):
        '''
        Adds the vegas lines to the dataframe.
        '''
        if self.season_type != 'Regular':
            return None
        elif self.year:
            if self.year == 2009:
                df = lines_2009
            elif self.year == 2010:
                df = lines_2010
            elif self.year == 2011:
                df = lines_2011
            elif self.year == 2012:
                df = lines_2012
            elif self.year == 2013:
                df = lines_2013
            elif self.year == 2014:
                df = lines_2014
            elif self.year == 2015:
                df = lines_2015
        else:
            df = all_lines
        if self.week:
            df = df[df['week'] == self.week]
        return df

    def _get_nfldb_df(self):
        '''
        Creates the correct position statistic dataframe.
        '''
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
        else:
            df = passing.append(rec).append(rush).append(te).append(dst)
        if self.season_type:
            df = df[df['season_type'] == self.season_type]
        if self.year:
            df = df[df['season_year'] == self.year]
        if self.week:
            df = df[df['week'] == self.week]
        return df

    def _merge_df(self):
        '''
        Merges the two dataframes.
        '''
        df1 = self._get_nfldb_df()
        df2 = self._sal_position()
        df = df1.merge(df2, on=['week', 'season_year', 'position', 'full_name'])
        return df

    def _add_lines(self):
        '''
        Add the spreads and lines to the dataframe.
        '''
        df1 = self._load_lines()
        df2 = self._merge_df()
        df = df1.merge(df2, on=['week', 'season_year', 'team'])
        return df


    def get_df(self):
        '''
        Allows the user to get the final dataframe.
        '''
        df = self._add_lines()
        df['points_per_dollar'] = (df['DK points'] / df['DK salary']) * 1000
        return df
