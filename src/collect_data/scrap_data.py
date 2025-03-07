import requests
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from pathlib import Path

station_ID = pd.read_csv("../../data/station_ID.csv", sep=",")
# drop les stations sans ID
station_ID = station_ID.dropna(subset = ["IDCJDW"])

# drop les nouvelles stations pour le moment
station_ID = station_ID.dropna(subset = ["Location"])

station_ID["IDCJDW"] = station_ID["IDCJDW"].astype(int).astype(str)
station_ID.head()

## time list 

year_list = ["2024","2025"]
month_list = {"2024": ["01","02","03","04","05","06","07","08","09","10","11","12"],
                "2025": ["01","02"]}

for year in year_list :
    for month in month_list[year] :
        for location in station_ID.Location2:
            print(location)
            id_location = station_ID.loc[(station_ID.Location2 == location) & 
                        (station_ID.Location == location),"IDCJDW"]
            
            if not(Path("../../data/scrapcsv/" + location + "_" + year + month + ".csv").exists()):


                url_csv = "https://reg.bom.gov.au/climate/dwo/" + \
                        year + month + "/text/IDCJDW" + str(id_location.iloc[0]) + "." + \
                            year + month + ".csv"
                
                response = requests.get(url_csv)

                if response.ok:
                    #print("url ok")
                    soup = BeautifulSoup(response.content, "html.parser")
                    lines = soup.prettify().split('\n')

                    for line in range(len(lines)) :
                        if lines[line][2:6] == "Date":
                            skiprows = line-1
                            print(skiprows)  
                            break
                         
                    datatmp = pd.read_csv(url_csv,
                                          skiprows= skiprows , 
                                          index_col=0,
                                          encoding = "ISO-8859-1")

                    if datatmp.columns[0] == "Date":
                        print("Date repérée -> enregistrement du csv ", location, "_",year+month)
                        datatmp.to_csv("../../data/scrapcsv/" + location + "_" + year + month + ".csv")


# Ajoute les données au dataset initial
import preprocess_scrapdata

preprocess_scrapdata.add_scrap_data(new_data_name = "weatherAUS_tuned")




# location = "BadgerysCreek"
# year = "2025"
# month = "01"

# print(location)
# id_location = station_ID.loc[(station_ID.Location2 == location) & 
#                              (station_ID.Location == location),"IDCJDW"]

# skiprows = station_ID.loc[(station_ID.Location2 == location) & 
#             (station_ID.Location == location),"skiprows"].values[0].astype(int)

# if not(Path("../../data/scrapcsv/" + location + "_" + year + month + ".csv").exists()):

#     r = requests.get(url_csv)
#     if r.ok:

#         url_csv = "https://reg.bom.gov.au/climate/dwo/" + \
#             year + month + "/text/IDCJDW" + str(id_location.iloc[0]) + "." + \
#                 year + month + ".csv"

#         datatmp = pd.read_csv(url_csv, 
#                         skiprows= skiprows , 
#                         index_col=1,
#                         encoding = "ISO-8859-1")
        
#         datatmp.to_csv("../../data/scrapcsv/" + location + "_" + year + month + ".csv")


# import urllib2  # the lib that handles the url stuff

# data = urllib2.urlopen(url_csv) # it's a file like object and works just like a file
# for line in data: # files are iterable
#     print(line)
# test sur les archives

# 2017 - 2020


# https://webarchive.nla.gov.au/awa/20170823140252/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/201707/html/IDCJDW2801.201707.shtml
# https://webarchive.nla.gov.au/awa/20180823140000/http://pandora.nla.gov.au/pan/44065/20180824-0000/www.bom.gov.au/climate/dwo/index.html
# https://webarchive.nla.gov.au/awa/20190823140000/http://pandora.nla.gov.au/pan/44065/20190824-0000/www.bom.gov.au/climate/dwo/index.html
# https://webarchive.nla.gov.au/awa/20200823140000/http://pandora.nla.gov.au/pan/44065/20200824-0000/www.bom.gov.au/climate/dwo/index.html
# https://web.archive.org.au/awa/20170823180131mp_/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

# https://web.archive.org.au/awa/20171408230237mp_/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/201707/html/IDCJDW2801.201707.shtml
# https://web.archive.org.au/awa/20170823140237mp_/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/201708/html/IDCJDW2801.201708.shtml
# https://web.archive.org.au/awa/20170823140237mp_/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/201708/html/IDCJDW2801.201708.shtml
# https://web.archive.org.au/awa/20170823140237mp_/http://pandora.nla.gov.au/pan/44065/20170824-0001/www.bom.gov.au/climate/dwo/201708/html/IDCJDW2801.201708.shtml

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

# tables = pd.read_csv("https://reg.bom.gov.au/climate/dwo/202502/text/IDCJDW2801.202502.csv",
#                      skiprows=7, index_col=1,
#                      encoding = "ISO-8859-1")

# tables.head()
# tables.tail()