import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn
import statsmodels.regression.linear_model as sm


defense_lst = ['Arizona Defense',
         'Los Angeles Defense',
         'Oakland Defense',
         'San Diego Defense',
         'San Francisco Defense',
         'Denver Defense',
         'Jacksonville Defense',
         'Miami Defense',
         'Tampa Bay Defense',
         'Atlanta Defense',
         'Chicago Defense',
         'Indianapolis Defense',
         'New Orleans Defense',
         'Baltimore Defense',
         'Washington Defense',
         'New Englnad Defense',
         'Detroit Defense',
         'Minnesota Defense',
         'Kansas City Defense',
         'New York G Defense',
         'New York J Defense',
         'Buffalo Defense',
         'Carolina Defense',
         'Cincinnati Defense',
         'Cleveland Defense',
         'Philadelphia Defense',
         'Pittsburgh Defense',
         'Tennessee Defense',
         'Dallas Defense',
         'Houston Defense',
         'Seattle Defense',
         'Green Bay Defense']

team_name_lst = ['Cardinals', 'Rams', 'Raiders', 'Chargers', '49ers', 'Broncos', 'Jaguars', 'Dolphins', 'Buccaneers', 'Falcons', 'Bear', 'Colts', 'Saints', 'Ravens', 'Redskins', 'Patriots', 'Lions', 'Vikings', 'Chiefs', 'Giants', 'Jets', 'Buffalo', 'Panthers', 'Bengals', 'Browns', 'Eagles', 'Steelers', 'Titans', 'Cowboys', 'Texans', 'Seahawks', 'Packers']



defense_dict = dict(itertools.izip(team_name_lst, defense_lst))


class dfAnalysis(object):



    def __init__(self, contest_file, point_file, defense_dict=defense_dict, delim_whitespace=False):

        self.defense_dict = defense_dict
        self.contest_file = contest_file
        self.point_file = point_file
        self.contest_df = self._load_contest_df(self.contest_file)
        self.point_df = self._load_point_df(self.point_file)
        self.lineup_df, self.percent_df = self._arrange_contest_data(self.contest_df)

        self.percent_df = self._clean_percent_df(self.percent_df, self.defense_dict)
        self.point_df = self._clean_point_df(self.point_df)
        self.point_percent = self._merge_point_percent_df(self.percent_df, self.point_df)

    def _load_contest_df(self, contest_file):

        '''
        Opens pandas dataframe with contest data. Outputs contest_df.
        '''


        contest_df = pd.read_csv(contest_file)
        return contest_df


    def _load_point_df(self, point_file, delimiter=';'):

        '''
        Open pandas dataframe with points data. Outputs point_df.
        '''


        point_df = pd.read_csv(point_file, delimiter=delimiter)
        return point_df


    def _arrange_contest_data(self, contest_df):
        '''
        Takes contest data as input, cleans data, and returns self.lineup_df and self.percent_df.
        '''
        contest_df['PercentDrafted'] = self.contest_df['%Drafted']
        contest_df.drop(labels=['EntryId', 'EntryName', 'TimeRemaining', 'Unnamed: 6', '%Drafted'], axis=1, inplace=True)
        lineup_df = contest_df[['Points', 'Lineup']]
        percent_df = contest_df[['Player', 'PercentDrafted']]
        return lineup_df, percent_df

    def _clean_percent_df(self, percent_df, defense_dict):
        self.percent_df.dropna(inplace=True)
        self.percent_df['PercentDrafted'] = self.percent_df['PercentDrafted'].apply(lambda x: x.strip('%'))
        self.percent_df['PercentDrafted'] = self.percent_df['PercentDrafted'].astype(float) * 0.01
        self.percent_df['Player'] = self.percent_df['Player'].apply(lambda x: x.strip())
        self.percent_df['Player'].replace(to_replace=self.defense_dict, inplace=True)
        return self.percent_df

    def _clean_point_df(self, point_df):
        self.point_df['Name'] = self.point_df['Name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))
        self.point_df.rename(index=str, columns={'Name': 'Player'}, inplace=True)
        return self.point_df

    def _merge_point_percent_df(self, percent_df, point_df):
        new_df = self.percent_df.merge(self.point_df, on='Player')
        new_df.drop(labels=['Week', 'Year', 'GID'], axis=1, inplace=True)
        new_df['h/a'] = new_df['h/a'].map({'h' : 0, 'a' : 1})
        new_df['points_per_dollar'] = (new_df['DK points'] / new_df['DK salary']) * 1000
        new_df['total_value_inversed'] = (1-new_df['PercentDrafted']) * new_df['points_per_dollar']
        return new_df


    def points_knapsack(self):
        budget = 50000
        current_team_salary = 0
        constraints = {
            'QB':1,
            'RB':2,
            'WR':3,
            'TE':1,
            'Def':1,
            'FLEX':1
            }

        counts = {
            'QB':0,
            'RB':0,
            'WR':0,
            'TE':0,
            'Def':0,
            'FLEX':0
            }

        pp = self.point_percent.sort('DK points', ascending=False)
        team = []

        for index, row  in pp.iterrows():
            if counts[row['Pos']] < constraints[row['Pos']] and current_team_salary + row['DK salary'] <= budget:
                team.append(row['Player'])
                counts[row['Pos']] = counts[row['Pos']] + 1
                current_team_salary += row['DK salary']
                continue
            if counts['FLEX'] < constraints['FLEX'] and current_team_salary + row['DK salary'] <= budget and row['Pos'] in ['RB','WR','TE']:
                team.append(row['Player'])
                counts['FLEX'] = counts['FLEX'] + 1
                current_team_salary += row['DK salary']

        return current_team_salary, team

    def plot_salary_vs_points(self):
        x = self.point_percent['DK salary']
        y = self.point_percent['DK points']
        labels = self.point_percent[self.point_percent['Player']]
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Salary')
        ax.set_ylabel('Points')
        ax.set_title('Salary vs Points')
        plt.show()

    def plot_salary_vs_ownership(self):
        x = self.point_percent['DK salary']
        y = self.point_percent['PercentDrafted']
        labels = self.point_percent['Player']
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Salary')
        ax.set_ylabel('Percent Owned')
        ax.set_title('Salary vs. Ownership')
        plt.show()

    def plot_points_vs_ownership(self):
        x = self.point_percent['DK points']
        y = self.point_percent['PercentDrafted']
        labels = self.point_percent['Player']
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Points')
        ax.set_ylabel('Percent Owned')
        ax.set_title('Points vs. Ownership')
        plt.show()
