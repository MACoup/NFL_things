from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests

request = requests.get('https://rotogrinders.com/team-stats/nfl-earned?site=draftkings&range=season')

soup = BeautifulSoup(request.content, 'html.parser')

table = soup.find('div', attrs={'class': 'rgtable'})
