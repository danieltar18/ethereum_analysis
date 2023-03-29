import requests
from bs4 import BeautifulSoup
import google_news_scraper_functions as custom_functions
from openpyxl import Workbook
import spacy
from datetime import date
import dateparser


workbook = Workbook()
worksheet = workbook.active
nlp = spacy.load('en_core_web_sm')
today = date.today()

worksheet.append(["date", "headline", "short_text", "link", "long_text", "sentiment"])

# Send a GET request to the Coindesk homepage
for i in range(1, 162):
    try:
        response = requests.get(f"https://www.coindesk.com/tag/ether/{i}/")

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        blocks = soup.find_all("div", class_="article-cardstyles__StyledWrapper-q1x8lc-0 hKWAzg article-card default")

        for block in blocks:
            # Find all headline elements on the page
            headline = block.find_all("h6")[-1]
            title = headline.text.strip()
            link = headline.find("a")["href"]
            short_text = block.find("span", class_="content-text").text.strip()
            date = block.find("span", class_="typography__StyledTypography-owin6q-0 fUOSEs").text.strip()
            article = custom_functions.download_and_translate_process(url="https://www.coindesk.com"+link)
            long_text, sentiment = custom_functions.sentiment_analysis(article, nlp)
            date = dateparser.parse(date)
            date = date.strftime("%Y-%m-%d")
            print(title, date)

            worksheet.append([date, title, short_text, link, long_text.text, sentiment])

        print("Jön a kövi oldal")
    except:
        continue

workbook.save(f"src/section2/eth_news/eth_scrapes/eth_results_{today}.xlsx")
