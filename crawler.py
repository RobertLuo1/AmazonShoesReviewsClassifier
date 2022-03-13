#-*- coding: utf-8 -*-
#@author: Zhuoyan Luo,Ruiming Chen

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import config
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


option = Options()
# option.add_argument('--headless')#设置无头请求
option.add_argument('--log-level=1')#解决google的报错
# option.add_argument('--ignore-certificate-error')
# option.add_argument('--ignore-ssl-error')
# option.add_experimental_option('excludeSwitches', ['enable-logging'])
option.add_experimental_option('excludeSwitches',['enable-automation']) #其中反反爬
s = Service(config.Chrome_PATH)
driver = webdriver.Chrome(service=s,options=option)#初始化driver


def getReview_2(driver,reviews,stars):
    countrys = ['美国', '加拿大', '印度', '英国']#只爬这几个国家的数据，尽量防止非英语状况
    check_xpath = '//div[@data-hook="review"]//span[@data-hook="review-date"]'
    check_list = driver.find_elements(By.XPATH, check_xpath)
    review_xpath = '//span[@class="a-size-base review-text review-text-content"]/span'
    reviews_elements = driver.find_elements(By.XPATH, review_xpath)
    if len(reviews_elements) == len(check_list):
        for i in range(len(check_list)):
            check_place = check_list[i].get_attribute('textContent')
            if not isSamepattern(check_place):
                break
            index_1 = check_place.index('在')
            index_2 = check_place.index('审')
            country = check_place[index_1 + 1:index_2]
            if country in countrys:
                if country == '美国':
                    star_xpath = '//div[@data-hook="review"]//div[@class="a-row"]/a/i/span'
                    stars_elements = driver.find_elements(By.XPATH, star_xpath)
                    star = stars_elements[i].get_attribute('textContent')[0]
                    if int(star) <= 3:
                        reviews.append(reviews_elements[i].text)
                        stars.append(star)
                else:
                    star_xpath = '//div[@data-hook="review"]//span[@class="a-icon-alt"]'
                    stars_elements = driver.find_elements(By.XPATH, star_xpath)
                    star = stars_elements[i].get_attribute('textContent')[0]
                    if int(star) <= 3:
                        reviews.append(reviews_elements[i].text)
                        stars.append(star)
    else:
        for i in range(len(check_list)):
            check_place = check_list[i].get_attribute('textContent')
            if not isSamepattern(check_place):
                break
            index_1 = check_place.index('在')
            index_2 = check_place.index('审')
            country = check_place[index_1 + 1:index_2]
            if country in countrys:
                if country == '美国':
                    short_review_xpath = '//div[@data-hook="review"]//a[@data-hook="review-title"]/span'
                    reviews_elements = driver.find_elements(By.XPATH, short_review_xpath)
                    if len(reviews_elements) != len(check_list):
                        break
                    star_xpath = '//div[@data-hook="review"]//div[@class="a-row"]/a/i/span'
                    stars_elements = driver.find_elements(By.XPATH, star_xpath)
                    star = stars_elements[i].get_attribute('textContent')[0]
                    if int(star) <= 3:
                        reviews.append(reviews_elements[i].text)
                        stars.append(star)
                else:
                    short_review_xpath = '//div[@data-hook="review"]//a[@data-hook="review-title"]/span'
                    reviews_elements = driver.find_elements(By.XPATH, short_review_xpath)
                    if len(reviews_elements) != len(check_list):
                        break
                    star_xpath = '//div[@data-hook="review"]//span[@class="a-icon-alt"]'
                    stars_elements = driver.find_elements(By.XPATH, star_xpath)
                    star = stars_elements[i].get_attribute('textContent')[0]
                    if int(star) <= 3:
                        reviews.append(reviews_elements[i].text)
                        stars.append(star)


def getReview_1(driver,reviews,stars):
    countrys = ['美国','加拿大','印度','英国']
    check_xpath = '//div[@data-hook="review"]//span[@data-hook="review-date"]'
    check_list = driver.find_elements(By.XPATH,check_xpath)
    review_xpath = '//span[@class="a-size-base review-text review-text-content"]/span'
    reviews_elements = driver.find_elements(By.XPATH, review_xpath)
    if len(reviews_elements) == len(check_list):
        for i in range(len(check_list)):
            check_place = check_list[i].get_attribute('textContent')
            if not isSamepattern(check_place):
                break
            index_1 = check_place.index('在')
            index_2 = check_place.index('审')
            country = check_place[index_1+1:index_2]
            if country in countrys:
                reviews.append(reviews_elements[i].text)
                if country == '美国':
                    star_xpath = '//div[@data-hook="review"]//div[@class="a-row"]/a/i/span'
                    stars_elements = driver.find_elements(By.XPATH,star_xpath)
                    stars.append(stars_elements[i].get_attribute('textContent')[0])
                else:
                    star_xpath = '//div[@data-hook="review"]//span[@class="a-icon-alt"]'
                    stars_elements = driver.find_elements(By.XPATH,star_xpath)
                    stars.append(stars_elements[i].get_attribute('textContent')[0])
    else:
        for i in range(len(check_list)):
            check_place = check_list[i].get_attribute('textContent')
            if not isSamepattern(check_place):
                break
            index_1 = check_place.index('在')
            index_2 = check_place.index('审')
            country = check_place[index_1+1:index_2]
            if country in countrys:
                if country == '美国':
                    short_review_xpath = '//div[@data-hook="review"]//a[@data-hook="review-title"]/span'
                    reviews_elements = driver.find_elements(By.XPATH, short_review_xpath)
                    if len(reviews_elements) != len(check_list):
                        break
                    reviews.append(reviews_elements[i].text)
                    star_xpath = '//div[@data-hook="review"]//div[@class="a-row"]/a/i/span'
                    stars_elements = driver.find_elements(By.XPATH,star_xpath)
                    stars.append(stars_elements[i].get_attribute('textContent')[0])
                else:
                    short_review_xpath = '//div[@data-hook="review"]//a[@data-hook="review-title"]/span'
                    reviews_elements = driver.find_elements(By.XPATH, short_review_xpath)
                    if len(reviews_elements) != len(check_list):
                        break
                    reviews.append(reviews_elements[i].text)
                    star_xpath = '//div[@data-hook="review"]//span[@class="a-icon-alt"]'
                    stars_elements = driver.find_elements(By.XPATH,star_xpath)
                    stars.append(stars_elements[i].get_attribute('textContent')[0])


def tocsv(reviews,stars,idx):
    df = pd.DataFrame()
    df['review'] = reviews
    df['star'] = stars
    df['label'] = [1 if int(star) > 3 else 0 for star in stars]
    df['link_idx'] = idx
    df.to_csv(config.data_file,index=False,encoding='utf-8')
    print('Finish')

def isInstance(driver,botton):
    exist = True
    try:
        link = driver.find_element(By.XPATH,botton)
    except:
        exist = False
    return exist

def isSamepattern(check_place):
    same = True
    try:
        index_1 = check_place.index('在')
        index_2 = check_place.index('审')
    except:
        same = False
    return same

def mainProcedure(driver):
    reviews = []
    stars = []
    count = 0
    index = 18
    while True:
        #link_xpath = '(//a[@class="a-link-normal a-text-normal"])'+'['+str(index)+']'
        link_xpath = '//a[@class="a-link-normal s-link-style a-text-normal"]'
        # "//a[@class="a-link-normal s-link-style a-text-normal"]"
        links = driver.find_elements(By.XPATH,link_xpath)
        if index == len(links):
            tocsv(reviews,stars,index)
            break
        link = links[index]
        link.click()
        time.sleep(2)
        count = 0
        search_all_botton_xpath = '//div[@class="a-row a-spacing-medium"]//a'
        if isInstance(driver,search_all_botton_xpath):
            driver.find_element(By.XPATH, search_all_botton_xpath).click()
            time.sleep(2)
            while count < 50:
                getReview_2(driver, reviews, stars)
                print(str(index)+f' the number of reviews is {len(reviews)}'+f' the number of stars is {len(stars)}')
                if len(reviews) < config.reviews_number:
                    nextpagebotton = '//div[@id="cm_cr-pagination_bar"]//li[@class="a-last"]/a'
                    if isInstance(driver, nextpagebotton):
                        driver.find_element(By.XPATH, nextpagebotton).click()
                        time.sleep(2)
                        count += 1
                    else:
                        break
                else:
                    tocsv(reviews, stars,index)
                    return
            driver.get(config.base_url+config.search_query)
            index+=1
            # driver.switch_to.window(base_window)
            time.sleep(2)
        else:
            driver.get(config.base_url + config.search_query)
            index+=1
            time.sleep(2)

driver.get(config.base_url+config.search_query)
driver.implicitly_wait(10)
# link_xpath = '//a[@class="a-link-normal a-text-normal"]'
# links = driver.find_elements(By.XPATH,link_xpath) #找到页面上所有的link
mainProcedure(driver)
driver.quit()
