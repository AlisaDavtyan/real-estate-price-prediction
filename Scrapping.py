import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet , LinearRegression
from  sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder ,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel


url = 'https://myrealty.am/hy/bnakaranneri-vacharq/Yerevan/7762?page='
data = []
for i in range(1,3):
    url_page = url + str(i)
    response = requests.get(url_page)
    response.text
    soup = BeautifulSoup(response.text, 'html.parser')


    for i in soup.find_all('div', class_='item-more mt-auto show-on-list d-flex'):
        link = i.a['href']
        house_response = requests.get(link)
        house_soup = BeautifulSoup(house_response.text, 'html.parser')

        house_id = house_soup.find('div',class_ = 'item-view-id').text.strip().split('ID ')[1]
        add_text = house_soup.find('div',class_ = 'col-auto item-view-address pl-0 mb-2 mt-1').text.strip().split(',')
        town ,distrinct, street = add_text[0] , add_text[1], add_text[2]
        price = house_soup.find('div',class_ = 'item-view-price').text.split()[0].strip()

        params_div = house_soup.find_all('ul', class_='item-view-list-params')
        house_div =  house_soup.find_all('div',class_ = 'col-12 d-flex justify-content-between justify-content-sm-start item-view-price-params')

        house_params0 = {}

        for param in params_div:
            params_items = param.find_all('li')  
            
            for item in params_items:
                label = item.find('label')  
                span = item.find('span')  
                
                label_text = label.text.strip() if label else "No label"
                span_text = span.text.strip() if span else "No value"
                house_params0[label_text] = span_text

            
        house_params1 = {}
        for house in house_soup.find_all("div", class_="col-12 d-flex justify-content-between justify-content-sm-start item-view-price-params"):
            for item in house.find_all("div"):
                label = item.find("label").text.strip()
                value = item.find("span").text.strip()
                house_params1[label] = value  
        

        house_data = {
            'ID' : house_id,
            'town' : town,
            'districts' : distrinct,
            'street' : street,
            'Գին' : price
        }

        house_data.update(house_params0)
        house_data.update(house_params1)

        data.append(house_data)

df= pd.DataFrame(data)