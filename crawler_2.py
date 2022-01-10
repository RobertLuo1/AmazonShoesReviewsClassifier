import config
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


option = Options()
# option.add_argument('--headless')#设置无头请求
option.add_argument('--log-level=1')#解决google的报错
# option.add_argument('--ignore-certificate-error')
# option.add_argument('--ignore-ssl-error')
# option.add_experimental_option('excludeSwitches', ['enable-logging'])
option.add_experimental_option('excludeSwitches',['enable-automation','enable-logging']) #其中反反爬
s = Service(config.Chrome_PATH)
driver = webdriver.Chrome(service=s,options=option)#初始化driver

driver.get('https://newids.seu.edu.cn/authserver/login?service=https://newids.seu.edu.cn/authserver/login2.jsp')
driver.implicitly_wait(10)
username_location_xpath = '//input[@id="username"]'
input_username = driver.find_element(By.XPATH,username_location_xpath)
input_username.send_keys('213191238')
password_xpath = '//input[@id="password"]'
input_password = driver.find_element(By.XPATH,password_xpath)
input_password.send_keys('zhuo2015*12*')
# button = driver.find_element(By.XPATH,'//button[@id="xsfw"]')
time.sleep(2)
js = 'document.getElementById("xsfw").click();'
driver.execute_script(js)
# webdriver.ActionChains(driver).move_to_element(button).
# button.click()
time.sleep(3)
driver.close()