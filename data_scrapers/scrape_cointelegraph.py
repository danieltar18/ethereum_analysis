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

"""
driver = webdriver.Chrome()

driver.get("https://cointelegraph.com/tags/ethereum")
time.sleep(2)
x = 0

for i in range(1, 3000):
    print("Current scroll index: {0}".format(i))
    if i == 1:
        x += 3000
    else:
        x += 500
    driver.execute_script(f"window.scrollTo(0, {x})")
    time.sleep(2)

with open("cointelegraph.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)

driver.close()
"""
workbook = Workbook()
worksheet = workbook.active
nlp = spacy.load('en_core_web_sm')
today = date.today()

worksheet.append(["date", "headline", "short_text", "link", "long_text", "sentiment"])

# Send a GET request to the Coindesk homepage
data = open('cointelegraph.html', 'r', encoding="utf-8")
soup = BeautifulSoup(data, 'html.parser')

# Parse the HTML content using BeautifulSoup
# soup = BeautifulSoup(response.content, "html.parser")


blocks = soup.find_all("li", class_="posts-listing__item")
for block in blocks:
    try:
        # Find all headline elements on the page
        headline = block.find("span", class_="post-card-inline__title").text.strip()
        link = block.find("a")["href"]
        short_text = block.find("p", class_="post-card-inline__text").text.strip()
        date = block.find("time", class_="post-card-inline__date").text.strip()
        article = custom_functions.download_and_translate_process(url="https://cointelegraph.com/"+link)
        print(article[0])
        long_text, sentiment = custom_functions.sentiment_analysis(article, nlp)
        print("Sentiment analízis lement")
        date = dateparser.parse(date)
        date = date.strftime("%Y-%m-%d")
        print(headline, date, sentiment)

        worksheet.append([date, headline, short_text, link, long_text.text, sentiment])
    except:
        continue

workbook.save(f"src/section2/eth_news/eth_scrapes/eth_results_cointelegraph.xlsx")
