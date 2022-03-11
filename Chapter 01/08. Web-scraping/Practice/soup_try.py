import requests
from bs4 import BeautifulSoup

page = requests.get('https://www.ilmeteo.it/meteo/Verona?refresh_ce')
print(page)

# #soup = BeautifulSoup(page.content, 'Html.parser')

# div = soup.find_all('div', id="current-conditions-body")

# temp = soup.find('p', class_="myfrecast-current-lrg")



