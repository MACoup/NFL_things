from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import request
import csv


# Rotogrinders Projection Scraping

def get_sals(week, year):
    DK_salaries = 'http://rotoguru1.com/cgi-bin/fyday.pl?week={}&year={}&game=dk&scsv=1'.format(week, year)
    r = requests.get(DK_salaries)
    html = r.content
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('pre').text
    with open('/Users/MACDaddy/fantasy_football/NFL_things/Draft_Kings/Data/player_scores/Week{}_Year{}_player_scores2.txt'.format(week, year), 'wb') as f:
        f.write(content)




if __name__ == '__main__':
    for y in range(2014, 2017):
        for w in range(1, 18):
            get_sals(w, y)
