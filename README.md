# Web Scraping with product search relevance using NLP, rules and image classification

## Introduction

Eccomerce web scraping provides insight into pricing data, market dynamics, and competitors’ practices. But often a particular search query for a product in an e-commerce website may give us non-relevant products in our data. 
For example, search queries `spider man car toy` and `spider man jacket` on the eCommerce website flipkart.com gave us some products that are unrelated to spiderman.


```python
from string import punctuation
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz

def clean_white_space(text):
    """
    to clean unwanted white space in text
    """
    if not text:
        return
    return " ".join(text.split())

def process_title(text):
    """
    to clean the text by lowercasing,
    and removing special characters and digits
    """
    text = text.lower()
    text = "".join([char for char in text if char not in punctuation and not char.isdigit()])
    return clean_white_space(text)

def string_match_with_fuzzy(text, threshold = 90):
    """
    to return true if text have a match with fuzzy partial ratio
    """
    text = process_title(text)
    # tokenize the text using nltk tokenizer
    text_list = word_tokenize(text)
    ratios = [fuzz.partial_ratio(text_, "spider") if len(text_) > 3 else 0 for text_ in text_list]
    if any([ratio_ >= threshold for ratio_ in ratios]):
        return True

def string_match_with_rules(text):
    """
    to return True if the following keywords are present
    in the text
    """
    text = process_title(text)
    naive_search = ["spiderman" in text, "spider man" in text, "spidey" in text, "spider" in text]
    if any(naive_search):
        return True
```
