#!/usr/bin/env python
# coding: utf-8

# In[54]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
import json
import os
import argparse
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait



def main(login_id,password,start,end):
    #login_id=input("id:")
    #password=input("pass:")
    login(login_id,password)
    #start=int(input("시작페이지:"))
    #end=int(input("끝페이지:"))
    crawl(start,end)
    

def login(login_id,password):
    driver.get('####')
    driver.implicitly_wait(10)
    driver.find_element_by_css_selector('body > app-root > menubar > div > ul > menubar-guest > span:nth-child(2)').click()
    driver.find_element_by_css_selector('body > app-root > app-modal > div > div > div > modal-contents > div > modal-login > input:nth-child(2)').send_keys(login_id)
    driver.find_element_by_css_selector('body > app-root > app-modal > div > div > div > modal-contents > div > modal-login > input:nth-child(3)').send_keys(password)
    driver.find_element_by_css_selector('body > app-root > app-modal > div > div > div > modal-contents > div > modal-login > button').click()
    print('login 완료^^')

def crawl(start,end):
    data={}
    base_url="####"
    i=start
    while i <= end:
        lec=[]
        url_id=str(i)
        url_tar=base_url+url_id
        print(url_id)
        try:
            driver.get(url_tar)
            wait.until(EC.url_to_be(url_tar))
            driver.implicitly_wait(10)
            context=driver.find_elements_by_css_selector("div.lecture-eval-content-context")      #강의평
            for j in context:
                lec.append(j.text)
                print(j.text)
            data[i]=lec
        except:
            data[i]=None
        i=i+1
    filename=str(start)+'_'+str(end)
    original_dir=r'C:\Users\limaries30\pythoncode\content'
    filedir=os.path.join(original_dir,filename)
    with open(filedir, 'a') as f:  # writing JSON object
        json.dump(data,f)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--login_id', help='id')
    parser.add_argument('--password', help='password')
    parser.add_argument('--start', help='start_page',type=int)
    parser.add_argument('--end', help='end_page',type=int)
     
    args = parser.parse_args()

    path='C:/Users/limaries30/Downloads/chromedriver_win32/chromedriver.exe' #chromedriver경로

    chrome_options = webdriver.ChromeOptions()      # 크롬 옵션 객체 생성
    chrome_options.add_argument('headless')        # headless 모드 설정
    chrome_options.add_argument("--disable-gpu")   # gpu 사용 안하도록 설정
    chrome_options.add_argument("lang=ko_KR")     # 한국어로 실행되도록 설정
    driver = webdriver.Chrome(path, options=chrome_options)
    wait = WebDriverWait(driver, 10)
    
        
    main(args.login_id,args.password,args.start,args.end)
        

