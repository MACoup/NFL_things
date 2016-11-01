import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

df = pd.read_table('Data/contest-standings-Sunday-Monday_Week3.txt', delimiter='\t')

df['PercentDrafted'] = df['%Drafted']

df.drop('%Drafted', axis=1, inplace=True)
#
perc_df = df[['Player', 'PercentDrafted']]
#
perc_df['PercentDrafted'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
#
# perc_df['PercentDrafted'] = perc_df['PercentDrafted'].astype(float) * 0.0001
#
# perc_df.dropna(inplace=True)
#
# df_scores_w3 = pd.read_csv('Week3_player_scores.txt', delimiter=';')




# if __name__=='__main__':

    # perc_df['Player'].replace(to_replace=['Cowboys ', 'Bears ', 'Saints ', 'Falcons '], value=['Dallas Defense', 'Chicago Defense', 'New Orleans Defense', 'Atlanta Defense'], inplace=True)
    # w3_night_scores = df_scores_w3.loc[df_scores_w3['Team'].isin(['dal', 'atl', 'nor', 'chi'])]
    #
    w3_night_scores['Name'] = w3_night_scores['Name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))
    #
    w3_night_scores.rename(index=str, columns={'Name': 'Player'}, inplace=True)
    #
    # new_df = w3_night_scores.merge(perc_df, on='Player')
    # sorted_new_df = new_df.sort('PercentDrafted', ascending=False)
    # new_df.drop(['Year', 'GID'], axis=1, inplace=True)
    #
    # keys = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
    # best_lineup = {}
