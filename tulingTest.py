#coding:utf-8
'''
from urllib import request
import json
from urllib.parse import quote
'''
#基于图灵机器人API的简单测试
import urllib.request
import urllib.parse
import json
import jieba
  
def getHtml(url):
    response = urllib.request.urlopen(url)
    page = response.read()
    page = page.decode('utf-8')
    return page
  
if __name__ == '__main__':  
    key = '791a60a250ef4b1caff07c9cbf656bd5'
    api = r'http://www.tuling123.com/openapi/api?key=' + key + '&info='
    while True:  
        info = input('我: ')
        #words = jieba.cut(info)
        #print(words)
        url = api + urllib.parse.quote(info)
        response = getHtml(url)  
        dic_json = json.loads(response)
        #results = jieba.cut(dic_json['text'])
        #print(results)
        print ('机器人: ' + dic_json['text'])