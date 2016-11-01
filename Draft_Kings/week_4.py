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


class dfAnalysis(object):
    def __init__(self):
        self.filename1 = filename1
        self.filename2 = filename2


    def load_contest_df(self, filename1, filename2):
        '''
        Opens two pandas dataframes with necessary data. Outputs contest_df and point_df.
        '''
        contest_df = pd.read_csv(filename1)
        return contest_df

    def load_point_df(self, filename2, delimiter=None):
        point_df = pd.read_csv(filename2, delimiter=delimiter)
        return point_df

    def arrange_contest_data(self, contest_df):
        '''
        Takes contest data as input, cleans data, and returns lineup_df and percent_df.
        '''
        contest_df['PercentDrafted'] = contest_df['%Drafted']
        contest_df.drop(labels=['EntryId', 'EntryName', 'TimeRemaining', 'Unnamed: 6', '%Drafted'], axis=1, inplace=True)
        lineup_df = contest_df[['Points', 'Lineup']]
        percent_df = contest_df[['Player', 'PercentDrafted']]
        return lineup_df, percent_df

    def clean_percent_df(self, percent_df, defense_dict):
        percent_df.dropna(inplace=True)
        percent_df['PercentDrafted'] = percent_df['PercentDrafted'].apply(lambda x: x.strip('%'))
        percent_df['PercentDrafted'] = percent_df['PercentDrafted'].astype(float) * 0.01
        percent_df['Player'] = percent_df['Player'].apply(lambda x: x.strip())
        percent_df['Player'].replace(to_replace=defense_dict, inplace=True)
        return percent_df

    def clean_point_df(self, point_df):
        point_df['Name'] = point_df['Name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))
        point_df.rename(index=str, columns={'Name': 'Player'}, inplace=True)
        return point_df

    # def clean_lineup_df(lineup_df):
    #     lineup_df.dropna(inplace=True)
    #     lineup_df['Lineup'] = lineup_df['Lineup'].map(lambda x: x.split())
    #     lineup_2 = pd.DataFrame(columns = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX'])
    #     idx = range(len(lineup_df['Lineup'][0])/3)
    #     idx = [i * 3 for i in idx]
    #     # for num in range(len(lineup_df['Lineup'])):
    #     #     for i in idx:
    #     #         lineup_2.ix[num, lineup_df['Lineup'][num][i]] = lineup_df['Lineup'][num][i+1] + ' ' + lineup_df['Lineup'][num][i+2]
    #     for num in range(len(lineup_df['Lineup'])):
    #         for i in idx:
    #             lineup_2[num]['QB'] = lineup_df['Lineup'][num][i+1] + ' ' + lineup_df['Lineup'][num][i+2]
    #
    #     return lineup_df, lineup_2

    # lineup_df.ix[0, lineup_df['Lineup'][0][3]] = lineup_df['Lineup'][0][0+1] + ' ' + lineup_df['Lineup'][0][0+2]

    def merge_data(self, percent_df, point_df):
        new_df = percent_df.merge(point_df, on='Player')
        return new_df

    def plot_salary_vs_points(self, point_percent):
        x = point_percent['DK salary']
        y = point_percent['DK points']
        labels = point_percent['Player']
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Salary')
        ax.set_ylabel('Points')
        ax.set_title('Salary vs Points')
        plt.show()

    def plot_salary_vs_ownership(self, point_percent):
        x = point_percent['DK salary']
        y = point_percent['PercentDrafted']
        labels = point_percent['Player']
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Salary')
        ax.set_ylabel('Percent Owned')
        ax.set_title('Salary vs. Ownership')
        plt.show()

    def plot_points_vs_ownership(self, point_percent):
        x = point_percent['DK points']
        y = point_percent['PercentDrafted']
        labels = point_percent['Player']
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for label, x, y in itertools.izip(labels, x, y):
            ax.annotate(label, xy = (x, y))
        ax.set_xlabel('Points')
        ax.set_ylabel('Percent Owned')
        ax.set_title('Points vs. Ownership')
        plt.show()




if __name__ == '__main__':
    filename1 = 'Data/contest-standings-sunday-monday-night-week4.csv'
    filename2 = 'Data/Week4_player_scores.txt'
    delimiter = ';'
    defense_dict = dict(itertools.izip(team_name_lst, defense_lst))

    week_4 = dfAnalysis()
    contest_df = week_4.load_contest_df(filename1, filename2)
    point_df = week_4.load_point_df(filename2, delimiter=delimiter)
    lineup_df, percent_df = week_4.arrange_contest_data(contest_df)
    percent_df = week_4.clean_percent_df(percent_df, defense_dict)
    point_df = week_4.clean_point_df(point_df)
    point_percent = week_4.merge_data(percent_df, point_df)
    # sal_vs_point = week_4.plot_salary_vs_points(point_percent)
    # sal_vs_own = week_4.plot_salary_vs_ownership(point_percent)
    # point_vs_own = week_4.plot_points_vs_ownership(point_percent)
    X = [point_percent['DK salary'], point_percent['PercentDrafted']]
    X = sm.add_constant(X)
    y = point_percent['DK points']
    sal_for_point_model = sm.OLS(y, X).fit()
