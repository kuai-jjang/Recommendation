from selenium import webdriver
import json
from collections import OrderedDict
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import argparse
import pandas as pd


def main(driver):

    classname=[]
    proff=[]
    star=[]
    sem=[]
    classtype=[]
    idx=0
    step=0
    for i in range(77935,79877,1):   #17-2 79876까지 하면됨
        step+=1
        star.append([])
        classname.append([])
        sem.append([])
        proff.append([])
        classtype.append([])
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
            classtype[idx].append(driver.find_element_by_xpath('/html/body/app-root/lecture-view/div/lecture-info/div/div[1]/div[2]/div[4]/table/tbody/tr[2]/td[1]').text)
            idx+=1
        except:
            continue
        if step%1000==0:   #backup
            df=pd.DataFrame(star,columns=['chulseok','grade','difficulty','load','achievement'])
            df['proff']=pd.DataFrame(proff)
            df['className']=pd.DataFrame(classname)
            df['sem']=pd.DataFrame(sem)
            df['classtype']=pd.DataFrame(classtype)
            df.to_csv('./stargazing_2017_2_2.csv',encoding='utf-8',index=False)
            with open('./here.txt','w') as f:
                f.write(str(i))

    df_2=pd.DataFrame(star,columns=['chulseok','grade','difficulty','load','achievement'])
    df_2['proff']=pd.DataFrame(proff)
    df_2['className']=pd.DataFrame(classname)
    df_2['sem']=pd.DataFrame(sem)
    df_2['classtype']=pd.DataFrame(classtype)
    df_2.to_csv('./stargazing_2017_2_3.csv',encoding='utf-8',index=False)
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str,required=True,help='web_id?')
    parser.add_argument('--pwd', type=str,required=True,help="wed_pwd?")

    args = parser.parse_args()


    driver = webdriver.Chrome(r'C:\Users\limaries30\tensorflow_code\chromedriver_win32\chromedriver.exe')
    driver.implicitly_wait(3)
    driver.get('https://klue.kr/')
    driver.implicitly_wait(3)
    driver.find_element_by_xpath('/html/body/app-root/menubar/div/ul/menubar-guest/span[2]').click()
    driver.implicitly_wait(3)
    driver.find_element_by_xpath('/html/body/app-root/app-modal/div/div/div/modal-contents/div/modal-login/input[1]').send_keys('limaries30')
    driver.implicitly_wait(3)
    driver.find_element_by_xpath('/html/body/app-root/app-modal/div/div/div/modal-contents/div/modal-login/input[2]').send_keys('genius0142')
    driver.implicitly_wait(3)
    driver.find_element_by_css_selector('body > app-root > app-modal > div > div > div > modal-contents > div > modal-login > button').click()
    driver.implicitly_wait(3)

    main(driver)


