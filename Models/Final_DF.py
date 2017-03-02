import pandas as pd
import numpy as np


NFL_lines = 'Data/NFL_lines_formatted/'
Player_data_root = 'Data/Positions_agg/'

class FinalDF(object):

    '''
    This creates the DataFrame that will be used in analysis. It combines the position dataframes with the salary and betting line dataframes.
    '''


    def __init__(self, season_type='Regular', position=None, year=None, week=None, load_lines=True, load_salaries=False):
        self.position = position
        self.year = year
        self.week = week
        self.season_type = season_type
        self.load_lines = load_lines
        self.load_salaries = load_salaries



    def _load_salaries(self):

        '''
        Formats salary dataframes. Creates a dataframe from each individual salary file, then appends them together.
        '''

        if self.load_salaries:
            df = pd.DataFrame()
            for y in range(2014, 2017):
                for w in range(1, 18):
                    new_df = pd.read_csv('Data/Player_DK_Salaries/Week{}_Year{}_player_scores2.txt'.format(w, y), delimiter=';')
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
        else:
            return None



    def _sal_position(self):

        '''
        Gets the correct position salary dataframe.
        '''

        if self.load_salaries:
            df = self._load_salaries()
            if self.position:
                df = df[df['position'] == self.position]
            else:
                df = df
            return df
        else:
            return None




    def _load_lines(self):

        '''
        Adds the vegas lines to the dataframe.
        '''


        if self.load_lines:
            if self.year:
                df = pd.read_csv(NFL_lines + 'lines_{}.csv'.format(self.year))
            else:
                df = pd.read_csv(NFL_lines + 'all_lines.csv')
            if self.week:
                df = df[df['week'] == self.week]
            return df
        else:
            return None




    def _get_nfldb_df(self):

        '''
        Creates the correct position statistic dataframe.
        '''


        if self.position:
            if self.position = 'QB_agg':
                df = pd.read_csv('Data/Positions_agg/passing_agg.csv')
            if self.position == 'QB':
                df = pd.read_csv(Player_data_root + 'passing_agg.csv')
            elif self.position == 'WR':
                df = pd.read_csv(Player_data_root + 'rec_agg.csv')
            elif self.position == 'RB':
                df = pd.read_csv(Player_data_root + 'rush_agg.csv')
            elif self.position == 'TE':
                df = pd.read_csv(Player_data_root + 'te_agg.csv')
            elif self.position == 'DST':
                df = pd.read_csv('Data/Position_dfs' + 'dst.csv')
        else:
            df = pd.read_csv(Player_data_root + 'all_stats.csv')
        if self.season_type:
            df = df[df['season_type'] == self.season_type]
        if self.year:
            df = df[df['season_year'] == self.year]
        if self.week:
            df = df[df['week'] == self.week]
        return df



    def _merge_df(self):

        '''
        Merges salaries with nfldb dataframes if both are True.
        '''


        df1 = self._get_nfldb_df()
        df2 = self._sal_position()
        df3 = self._load_lines()
        if self.load_salaries and self.load_lines:
            df4 = df1.merge(df2, on=['week', 'season_year', 'position', 'full_name'])
            return df4.merge(df3, on=['week', 'season_year', 'team'])
        elif self.load_lines:
            return df1.merge(df3, on=['week', 'season_year', 'team'])
        elif self.load_salaries:
            return df1.merge(df2, on=['week', 'season_year', 'position', 'full_name'])
        else:
            return df1



    def get_df(self):


        '''
        Allows the user to get the final dataframe.
        '''


        df = self._merge_df()
        if self.load_salaries:
            df['points_per_dollar'] = (df['DK points'] / df['DK salary']) * 1000
        return df
