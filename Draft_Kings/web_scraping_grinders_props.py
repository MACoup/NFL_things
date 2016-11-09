from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from StringIO import StringIO


# Rotogrinders Projection Scraping
QB = 'https://rotogrinders.com/projected-stats/nfl-qb.csv?site=draftkings'
RB = 'https://rotogrinders.com/projected-stats/nfl-rb.csv?site=draftkings'
WR = 'https://rotogrinders.com/projected-stats/nfl-wr.csv?site=draftkings'
TE = 'https://rotogrinders.com/projected-stats/nfl-te.csv?site=draftkings'
DST = 'https://rotogrinders.com/projected-stats/nfl-defense.csv?site=draftkings'


def get_soup(url):
    r = requests.get(url)
    html = r.content
    soup = BeautifulSoup(html, 'html.parser')
    return soup

qb_grinders = get_soup(QB)
rb_grinders = get_soup(RB)
wr_grinders = get_soup(WR)
te_grinders = get_soup(TE)
dst_grinders = get_soup(DST)

def rotogrinders_soup(soup):
    data = StringIO(soup)
    cols = ['player', 'salary', 'team', 'position', 'opp', 'ceiling', 'floor', 'proj_points']
    df = pd.read_csv(data, header=None, names=cols)
    return df

qb = rotogrinders_soup(qb_grinders)
rb = rotogrinders_soup(rb_grinders)
wr = rotogrinders_soup(wr_grinders)
te = rotogrinders_soup(te_grinders)
defst = rotogrinders_soup(dst_grinders)

if __name__ == '__main__':
    qb.to_csv('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/qb_grinders_week10.csv')
    rb.to_csv('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/rb_grinders_week10.csv')
    wr.to_csv('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/wr_grinders_week10.csv')
    te.to_csv('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/te_grinders_week10.csv')
    defst.to_csv('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/defst_grinders_week10.csv')
