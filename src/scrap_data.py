import requests
import requests
import pandas as pd
from bs4 import BeautifulSoup

# #r = requests.get("https://reg.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml")
# print(r)
# print(r.content)

# soup = BeautifulSoup(r.content, 'html.parser')
# print(soup.prettify())
# content = soup.find_all('table')
# print(content)

# tables = pd.read_html("https://reg.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml")
# print(tables[0])

# table_test = pd.DataFrame(tables[0])
# # retire les statistiques
# table_test = table_test.iloc[:-4,]
# table_test.tail()

# # retire
# # check date 

# table_test["Date"]

tables = pd.read_csv("https://reg.bom.gov.au/climate/dwo/202502/text/IDCJDW2801.202502.csv",
                     skiprows=7, index_col=1,
                     encoding = "ISO-8859-1")

tables.head()
tables.tail()