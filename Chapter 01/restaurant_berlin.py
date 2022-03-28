from unicodedata import name
import bs4 as bs
import urllib.request as url
import pandas as pd
from datetime import timedelta, date
import re
import requests
from bs4 import BeautifulSoup



source = url.urlopen('https://www.yelp.co.uk/search?cflt=restaurants&find_loc=Berlin%2C+Germany')

page_soup = bs.BeautifulSoup(source, 'html.parser')


mains = page_soup.find_all("div", {"class": "css-1m051bw"})




soup = []
page = requests.get(f'https://www.yelp.co.uk/search?cflt=restaurants&find_loc=Berlin&start=0')
print(page)
soup.append(BeautifulSoup(page.content,'html.parser'))


business_name = []

for item in soup:
    business = item.find_all('a',class_="css-1m051bw")



soup = {}

for start_num in range(0, 230, 10):
    url = f"https://www.yelp.co.uk/search?cflt=restaurants&find_loc=Berlin&start={start_num}"
    yelp_berlin = requests.get(url)
    print(yelp_berlin)
    page = BeautifulSoup(yelp_berlin.content, 'html.parser')
    soup[start_num] = page

soup_copy = soup.copy()
pd.DataFrame([soup.copy]).to_csv("restaurant")









