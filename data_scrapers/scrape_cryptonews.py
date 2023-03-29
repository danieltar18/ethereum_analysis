import time

import requests
from bs4 import BeautifulSoup
import google_news_scraper_functions as custom_functions
from openpyxl import Workbook
import spacy
from datetime import date
import dateparser
from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Chrome()

driver.get("https://cryptonews.com/news/ethereum-news/")
time.sleep(2)
x = 0

for i in range(1, 150):
    try:
        submit = driver.find_elements(By.CLASS_NAME, "cookieConsent__Button")[-1].click()
    except:
        pass

    try:
        print("Current scroll index: {0}".format(i))
        element = driver.find_element(By.ID, "load_more")
        driver.execute_script("arguments[0].scrollIntoView(false)", element)
        time.sleep(1)
        element.click()
        time.sleep(1)
    except:
        time.sleep(0.5)
        driver.find_element(By.XPATH, '//*[@id="newsletter-modal"]/div/div[1]').click()

with open("cryptonews.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)

driver.close()

workbook = Workbook()
worksheet = workbook.active
nlp = spacy.load('en_core_web_sm')
today = date.today()

worksheet.append(["date", "headline", "short_text", "link", "long_text", "sentiment"])

# Send a GET request to the Coindesk homepage
data = open('cryptonews.html', 'r', encoding="utf-8")
soup = BeautifulSoup(data, 'html.parser')

# Parse the HTML content using BeautifulSoup
# soup = BeautifulSoup(response.content, "html.parser")

blocks = soup.find_all("article", class_="mb-30")
for block in blocks:
    # Find all headline elements on the page
    headline = block.find("h4").text.strip()
    link = block.find("a")["href"]
    #short_text = block.find("p", class_="post-card-inline__text").text.strip()
    date = block.find("time", class_="post-card-inline__date").text.strip()
    article = custom_functions.download_and_translate_process(url="https://cointelegraph.com/"+link)
    long_text, sentiment = custom_functions.sentiment_analysis(article, nlp)
    date = dateparser.parse(date)
    date = date.strftime("%Y-%m-%d")
    print(headline, date, sentiment)

    worksheet.append([date, headline, short_text, link, long_text, sentiment])

print("Jön a kövi oldal")


workbook.save(f"src/section2/eth_news/eth_scrapes/eth_results_cointelegraph.xlsx")
