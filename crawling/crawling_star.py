from selenium import webdriver
import json
from collections import OrderedDict
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd




def main():
    
    classname=[]
    proff=[]
    star=[]
    sem=[]
    idx=0
    for i in range(91129,91244,1):
        star.append([])
        classname.append([])
        sem.append([])
        proff.append([])
        url="https://klue.kr/lecture/"+str(i)
        driver.get(url)
        driver.implicitly_wait(3)
        try:
            for j in range(5):
                buss="#lecture_"+str(i)+"-g > text:nth-child("+str(j+12)+")"
                watcha=driver.find_element_by_css_selector(buss).text
                
                star[idx].append(watcha)
            classname[idx].append(driver.find_element_by_xpath('/html/body/app-root/lecture-view/div/lecture-info/div/div[1]/div[2]/div[1]').text)
            sem[idx].append(driver.find_element_by_xpath('/html/body/app-root/lecture-view/div/lecture-info/div/div[1]/div[2]/div[4]/table/tbody/tr[1]/td[1]').text)
            proff[idx].append(driver.find_element_by_xpath('/html/body/app-root/lecture-view/div/lecture-info/div/div[1]/div[2]/div[2]').text)
            idx+=1
        except:
            continue

    df=pd.DataFrame(star,columns=['chulseok','grade','difficulty','load','achievement'])
    df['proff']=pd.DataFrame(proff)
    df['className']=pd.DataFrame(classname)
    df['sem']=pd.DataFrame(sem)
    df.to_csv('./stargazing.csv',encoding='utf-8',index=False)
    

if __name__=="__main__":

    driver = webdriver.Chrome(r'C:\Users\limaries30\tensorflow_code\chromedriver_win32\chromedriver.exe')
    driver.implicitly_wait(3)
    driver.get('###')
    driver.implicitly_wait(3)
    driver.find_element_by_xpath('/html/body/app-root/menubar/div/ul/menubar-guest/span[2]').click()
    driver.find_element_by_xpath('/html/body/app-root/app-modal/div/div/div/modal-contents/div/modal-login/input[1]').send_keys('###')
    driver.find_element_by_xpath('/html/body/app-root/app-modal/div/div/div/modal-contents/div/modal-login/input[2]').send_keys('###')
    driver.find_element_by_css_selector('body > app-root > app-modal > div > div > div > modal-contents > div > modal-login > button').click()


