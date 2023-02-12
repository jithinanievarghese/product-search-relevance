# Web Scraping with product search relevance using NLP, rules and image classification

## Introduction

Eccomerce web scraping provides insight into pricing data, market dynamics, and competitors’ practices. But often a particular search query for a product in an e-commerce website may give us non-relevant products in our data. 

For example, search queries `spider man car toy` and `spider man jacket` on the eCommerce website [flipkart.com](https://www.flipkart.com/search?q=spider%20man%20jacket&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off) gave us some products that are unrelated to spiderman.  
<img src="https://user-images.githubusercontent.com/78400305/218292800-df0aefcb-dcf2-4011-b90d-8ab2e3d92904.png" width="500" height="300"><img src="https://user-images.githubusercontent.com/78400305/218292673-237924c3-b61b-4668-aaeb-c77a95c43fc0.png" width="500" height="300">

In a real-world scenario, we won't be using 3-4 keywords in our search, but more than 50 or 100 or even more. 



So manually identifying the unwanted products in a large amount of data is a tedious task. Also, we don't scrape data once but on a daily, weekly, or monthly basis for price monitoring and to identify the new products that are launched by our competitors.

## Approach to the Problem

The approach to the problem really depends on the ROI of the project. 
One  cost-effective way is to use a rule-based approach or use string matching algorithms like [Levenshtein Distance](https://medium.com/analytics-vidhya/fuzzy-matching-in-python-2def168dee4a) to identify the relevant products from the product title.

[Link to the notebook]



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

def string_match_with_fuzzy(text, threshold=90):
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
        
title = "SEMAPHORE bobblehead Toys Action Figure and Car Dashboard Interior Accessories(SPIDERMAN) Compatible with Hyundai Verna"
start = time()
print("string match with rules , title match:", string_match_with_rules(title))
print(f"time taken to process {time()-start} seconds\n")
start = time()
print("string match with Levenshtein Distance (partial ratio), title match:", string_match_with_fuzzy(title))
print(f"time taken to process {time()-start} seconds") 
```
#### Output
```python
string match with rules , title match: True
time taken to process 0.00020599365234375 seconds

string match with Levenshtein Distance (partial ratio), title match: True
time taken to process 0.0004353523254394531 seconds

```

But the major drawback with this approach is the relevant products with titles that don't have the keyword related to our search query will be missed from our final data.

Example for the following [product](https://www.flipkart.com/viaan-boys-cartoon-superhero-cotton-blend-t-shirt/p/itm4e53e909cfe2f?pid=KTBFTMVYQECWS8RG&lid=LSTKTBFTMVYQECWS8RG6PPDJ4&marketplace=FLIPKART&q=spiderman+t+shirt&store=clo%2Fash%2Fank%2Fpgi&srno=s_1_18&otracker=search&otracker1=search&fm=Search&iid=6aad40ff-eaa0-444f-8551-64fd3b1fda33.KTBFTMVYQECWS8RG.SEARCH&ppt=sp&ppn=sp&ssid=ozsx5aobzk0000001676176907270&qH=2209dc77098ed2d6)  `Boys Cartoon/Superhero Cotton Blend T Shirt  (White, Pack of 1)` we can’t identify the title as a relevant one since it doesn't have any term related to spiderman. So the only way to identify them is from their product image.

<img src="https://user-images.githubusercontent.com/78400305/218293631-c3622e30-4feb-4a24-b4c2-60d8df7ffa75.png" width="500" height="300"><img src="https://user-images.githubusercontent.com/78400305/218293636-5f9d236f-7531-4947-9c9e-aaa66b627ba4.png" width="500" height="300">


