#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:26:54 2023

@author: MacBookPro-II
"""
import requests
# API_KEY = 'trnsl.1.1.20230113T181150Z.89643c13965318fa.9020c272e0a6fc8fb0ce39558375e0e740c23216'
API_KEY = 'trnsl.1.1.20230113T181150Z.89643c13965318fa.9020c272e0a6fc8fb0ce39558375e0e740c2321678'


url = 'https://translate.yandex.net/api/v1.5/tr.json/translate'
# res = requests.get(url)
"""if res:
    print('Response OK')
else:
    print('Response Failed')"""



"""print(res)
print(res.headers)
print(res.text)"""

params = dict(key=API_KEY, text='Goodbye', lang='en-es')
res = requests.get(url, params=params)
print(res.text)

print(res.headers)
json = res.json()
print("json:", json)
print(json['text'])
print(json['text'][0])
print(res.status_code)
