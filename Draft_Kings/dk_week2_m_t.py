import pandas as pd
import numpy as np


sals = pd.read_csv('/Users/MACDaddy/Desktop/DKSalaries.csv')

sals.drop('GameInfo', axis=1, inplace=True)

# Create new DFs based on position
sals_QB = sals[sals['Position'] == 'QB']
sals_WR = sals[sals['Position'] == 'WR']
sals_RB = sals[sals['Position'] == 'RB']
sals_TE = sals[sals['Position'] == 'TE']
sals_DST = sals[sals['Position'] == 'DST']

# Drop select players
sals_QB = sals_QB[sals_QB['AvgPointsPerGame'] != 0]

df = pd.read_csv('/')
df.iloc['QB', 0] = df[df['Name + ID'].str.contains('Carson')]['Name + ID']

if __name__ == '__main__':
    print 'Sals: ', sals.head()
    print ''
    print 'QB:' , sals_QB
    print ''
    print 'WR: ', sals_WR
    print ''
    print 'RB: ', sals_RB
    print ''
    print 'TE: ', sals_TE
    print ''
    print 'DST: ', sals_DST
