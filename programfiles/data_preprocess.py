import pandas as pd
import re
from bs4 import BeautifulSoup

reviews = pd.read_csv('../datafiles/MH_12star_reviews.csv')

def pre_process(text):
  text = BeautifulSoup(str(text)).get_text()
  text = re.sub("[^a-zA-Z]", " ", text)
  text = text.lower()
  tokens = text.split()
  return " ".join(tokens)

reviews['clean_content'] = reviews['content'].apply(pre_process)
reviews.to_csv('../datafiles/MH_12star_reviews.csv')