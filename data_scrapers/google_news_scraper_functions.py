from newspaper import Article
import spacy
from string import punctuation
from heapq import nlargest
from googletrans import Translator
import time
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import math
from spacytextblob.spacytextblob import SpacyTextBlob


STOP_WORDS_HU = stopwords.words('hungarian')

translator = Translator()

def download_and_translate_process(url):
    try:
        url = url
        article = Article(url)
        article.download()
        article.parse()
        time.sleep(1)
        return article.text
    except:
        return ""

def summarize(text, per, nlp, language="en"):
    if len(text) > 0:
        if language == "hu":
            doc= nlp(text)
            tokens=[token.text for token in doc]
            word_frequencies = {}
            for word in doc:
                if word.text.lower() not in list(STOP_WORDS_HU):
                    if word.text.lower() not in punctuation:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] += 1
            #print("Word frequencies: {}".format(word_frequencies))
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word]=word_frequencies[word]/max_frequency
            sentence_tokens= [sent for sent in doc.sents]
            sentence_scores = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text.lower() in word_frequencies.keys():
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent]=word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent]+=word_frequencies[word.text.lower()]
            #print("Sentence score {}".format(sentence_scores))
            select_length=int(math.ceil(len(sentence_tokens)*per))
            #print("Select length: {}".format(select_length))
            summary=nlargest(select_length, sentence_scores, key=sentence_scores.get)
            #print("Summary {}".format(summary))
            final_summary=[word.text for word in summary]
            #print(final_summary)
            summary=''.join(final_summary)
            return summary.replace("\n", "")

        else:
            doc = nlp(text)
            tokens = [token.text for token in doc]
            word_frequencies = {}
            for word in doc:
                if word.text.lower() not in list(STOP_WORDS):
                    if word.text.lower() not in punctuation:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] += 1
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency
            sentence_tokens = [sent for sent in doc.sents]
            sentence_scores = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text.lower() in word_frequencies.keys():
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]
            select_length = int(len(sentence_tokens) * per)
            summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
            final_summary = [word.text for word in summary]
            summary = ''.join(final_summary)
            return summary.replace("\n", "")
    else:
        return ""

def sentiment_analysis(text, nlp):
    if len(text) > 0:
        text = text
        nlp.add_pipe('spacytextblob')
        try:
            translator = Translator()
            text = str(translator.translate(text))

            doc = nlp(text)

            nlp.remove_pipe('spacytextblob')
        except:
            pass

        return text, doc._.blob.polarity
    else:
        return  text, 0

def max_columns_width(worksheet):
    worksheet = worksheet

    for col in worksheet.columns:
        max_length = 0
        column = col[0].column_letter  # Get the column name
        for cell in col:
            try:  # Necessary to avoid error on empty cells
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column].width = adjusted_width