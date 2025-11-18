# app/utils/text_cleaning.py
import re
import string
import html

def clean_text(text: str) -> str:
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\\n|\\r|\n|\r', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
