from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from openpyxl import Workbook
from datetime import date
import google_news_scraper_functions
import spacy
import dateparser


today = date.today()
workbook = Workbook()
worksheet = workbook.active

search_key = "ethereum"
language = "en"

if language == "hu":
    nlp = spacy.load("hu_core_news_lg")
else:
    nlp = spacy.load('en_core_web_sm')

driver = webdriver.Chrome()
driver.get('https://www.google.com/search?q=ethereum&hl=en&tbas=0&biw=1036&bih=666&source=lnt&tbs=sbd%3A1%2Ccdr%3A1%2Ccd_min%3A1%2F9%2F2017%2Ccd_max%3A8%2F3%2F2023&tbm=nws')
time.sleep(1)
driver.find_element(By.XPATH,
                    '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button').click()
time.sleep(1)

kovetkezo_gomb = "valami"

worksheet.append(["date", "post_header", "link", "text", "sentiment"])
postok = 1

while kovetkezo_gomb is not None or len(postok) > 0:
    time.sleep(1)
    postok = driver.find_elements(By.CLASS_NAME, 'SoaBEf')
    for element in postok:
        date = element.find_element(By.CLASS_NAME, 'YsWzw').text
        date = dateparser.parse(date)
        date = date.strftime("%Y-%m-%d")
        post_header = element.find_element(By.CLASS_NAME, 'MBeuO').text
        link = element.find_element(By.CLASS_NAME, 'WlydOe').get_attribute('href')
        time.sleep(1)
        try:
            article = google_news_scraper_functions.download_and_translate_process(link)
            text, sentiment = google_news_scraper_functions.sentiment_analysis(article, nlp)
        except:
            sentiment = 0
        worksheet.append([date, post_header, link, text.text, sentiment])

    #kovetkezo_gomb = None
    try:
        kovetkezo_gomb = driver.find_element(By.XPATH, '//*[@id="pnnext"]/span[2]')
        time.sleep(1)
        kovetkezo_gomb.click()
        print("Jön a kövi oldal")
    except:
        break
    #break

#timi_functions.max_columns_width(worksheet=worksheet)

workbook.save("src/section2/google_news/eth_scrapes/{0}_result_{1}.xlsx".format(search_key, today))
