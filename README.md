# Web Scraping with product search relevance using NLP, rules and image classification

## Introduction

Eccomerce web scraping provides insight into pricing data, market dynamics, and competitors’ practices. But often a particular search query for a product in an e-commerce website may give us non-relevant products in our data. 

For example, search queries `spider man car toy` and `spider man jacket` on the eCommerce website [flipkart.com](https://www.flipkart.com/search?q=spider%20man%20jacket&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off) gave us some products that are unrelated to spiderman.  
<img src="https://user-images.githubusercontent.com/78400305/218292800-df0aefcb-dcf2-4011-b90d-8ab2e3d92904.png" width="500" height="300"><img src="https://user-images.githubusercontent.com/78400305/218292673-237924c3-b61b-4668-aaeb-c77a95c43fc0.png" width="500" height="300">

In a real-world scenario, we won't be using 3-4 keywords in our search, but more than 50 or 100 or even more. 



So manually identifying the unwanted products in a large amount of data is a tedious task. Also, we don't scrape data once but on a daily, weekly, or monthly basis for price monitoring and to identify the new products that are launched by our competitors.

## Approach to the Problem

### Using NLP or rules
There are several approaches to the problem, but choosing one over another or a combination really depends on the ROI of the project
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

def string_match_with_fuzz(text, threshold=90):
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

### Image classification
So with the help of image classification, we can identify the products. But for that, we need to gather the images of products along with other product page requests in the data gathering. 
If we have 50k products, we need to send an additional 50k image requests to the image URLs of products which adds up to the cost of data gathering.

Also training a model, its optimization, developer cost, and deployment also add up to the entire cost. So implementing image classification really depends on our ROI. 
So it is our call to whether we need to implement the image classification model for finding the product images. 
If that approach really adds value to our business requirements, then we should implement it.

For now, we have gathered data for 5 search queries "spider man car toy", "spider man mug", "spider man jacket", "spider man hoodies" and "spiderman t shirt". The total unique products for all search queries at the time of scraping was 2233. 
We are considering this problem as a binary image classification problem and our performance metrics will be f1-score and confusion matrix over accuracy since we need to know how well our model is going to identify the relevant products.

But do we have any labeled training data? 
Yes, we have partial training data 😃💡.
In the entire process of finding the relevant title using a string matching algorithm, we got around 1175 out of 2233 products as relevant ones, this is a form of weak supervision to label the data.
```python
start = time()
print("total no of products:", df.shape[0])
df["relevant_products_fuzz_match"] = df.title.apply(lambda x: string_match_with_fuzz(x))
print(f"time for processing: {time()-start} seconds", )
relevant_products = df[~df.relevant_products_fuzz_match.isnull()]
print("no of products identified:", relevant_products.shape[0])
relevant_products = relevant_products.drop_duplicates(subset=['image_url'])
relevant_products.shape
print("no of relevant product images after removing duplicated image urls:", relevant_products.shape[0])
```
#### Output
```python
total no of products: 2233
time for processing: 0.20037364959716797 seconds
no of products identified: 1175
no of relevant product images after removing duplicated image urls: 1124
```

Some of the products had the same image URLs, so after removing the duplicates we had around known 1124 relevant product images and we labeled them as class 1 or Target 1. For class 2 or Target 0 data, we can make use of open-source datasets that are related to our domain i.e clothing, toys, and mugs. 
Target 0 images mean data that shouldn't have a spiderman image. We need Images of T-shirts, hoodies, jackets, mugs, and toys that are not related to spiderman. We were able to source some clothing data from Kaggle. 

I am extremely thankful to the following kaggle dataset contributors.  
From their datasets, we were able to gather Target 0 images for our training data
1. https://www.kaggle.com/datasets/sunnykusawa/tshirts
2. https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl
3. https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

We ignored the available toys data in the 3rd dataset because there may be a chance of spiderman toys in that dataset and it will affect our model performance.





