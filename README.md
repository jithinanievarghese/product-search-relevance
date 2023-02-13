# Web Scraping with product search relevance using NLP, rules and image classification

## Project Flow

<p align="center">
  <img width="450" height="300" src="https://user-images.githubusercontent.com/78400305/218310437-94a0bc78-514a-42ce-a8e3-b4aa29bf2f2d.png">
</p>

## Introduction

E-commerce web scraping provides valuable insights into pricing data, market dynamics, and competitors practices.   
But often a particular search query in an e-commerce website may give us non-relevant products in our data. 

For example, search queries `spider man car toy` and `spider man jacket` on the e-commerce website [flipkart.com](https://www.flipkart.com/search?q=spider%20man%20jacket&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off) gave us some products that are not related to spiderman car, toy or jacket.  

<img src="https://user-images.githubusercontent.com/78400305/218292800-df0aefcb-dcf2-4011-b90d-8ab2e3d92904.png" width="500" height="300"><img src="https://user-images.githubusercontent.com/78400305/218292673-237924c3-b61b-4668-aaeb-c77a95c43fc0.png" width="500" height="300">

- In a real-world scenario, we won't be using 3-4 keywords in e-commerce scraping, but more than 50 or 100 or even more. 
- So manually identifying the unwanted products in a large amount of data is a tedious task. 
- These kind of non-relevant data are outliers in our gathered data and they dont add any value to our business requirements or data analysis
- Also, we don't scrape data once, but on a daily, weekly, or monthly basis for price monitoring or to identify the new products that are launched by our competitors. 
- So what if we need an automated solution for this i.e we can identify the unwanted products during the scraping and provide a clean dataset to our price monitoring or data analysis platform.


## Approach to the Problem

### Using NLP or rules
There are several approaches to the problem, but choosing one over another or a combination really depends on the ROI of the project.
One  cost-effective way is to use a rule-based approach or use string matching algorithms like [Levenshtein Distance](https://medium.com/analytics-vidhya/fuzzy-matching-in-python-2def168dee4a) to identify the relevant products from the product title.

##### Code: [Data pre-processing](https://github.com/jithinanievarghese/flipkart_scraper_scrapy/blob/main/flipkart_scraper/flipkart_scraper/data%20preprocesing%20product%20search%20relevance.ipynb)



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

But the major drawback with this approach is that the relevant products with titles that don't have the keyword related to our search query will be missed from our final data.

Example for the following [product](https://www.flipkart.com/viaan-boys-cartoon-superhero-cotton-blend-t-shirt/p/itm4e53e909cfe2f?pid=KTBFTMVYQECWS8RG&lid=LSTKTBFTMVYQECWS8RG6PPDJ4&marketplace=FLIPKART&q=spiderman+t+shirt&store=clo%2Fash%2Fank%2Fpgi&srno=s_1_18&otracker=search&otracker1=search&fm=Search&iid=6aad40ff-eaa0-444f-8551-64fd3b1fda33.KTBFTMVYQECWS8RG.SEARCH&ppt=sp&ppn=sp&ssid=ozsx5aobzk0000001676176907270&qH=2209dc77098ed2d6)  `Boys Cartoon/Superhero Cotton Blend T Shirt  (White, Pack of 1)` we can’t identify the title as a relevant one since it doesn't have any term related to spiderman. So the only way to identify them is from their product image.

<img src="https://user-images.githubusercontent.com/78400305/218293631-c3622e30-4feb-4a24-b4c2-60d8df7ffa75.png" width="500" height="300"><img src="https://user-images.githubusercontent.com/78400305/218293636-5f9d236f-7531-4947-9c9e-aaa66b627ba4.png" width="500" height="300">

### Image classification
So with the help of image classification, we can identify the products. But for that, we need to gather the images of products along with other product page requests in the data gathering. 
If we have 50k products, we need to send an additional 50k image requests to the image URLs of products which adds up to the cost of data gathering (We can reduce the image requests, discussed later in [Deployment, Scrape Flow and Conclusion](https://github.com/jithinanievarghese/product-search-relevance/edit/main/README.md#deployment-flow-and-conclusion)).

- Also training a model, its optimization, developer cost, and deployment also add up to the entire cost. So implementing image classification really depends on our ROI. 
- So it is our call to decide whether we need to implement the image classification model for finding the product images. 
- If that approach really adds value to our business requirements, then we should implement it.

For now, we have gathered data for 5 search queries `spider man car toy`, `spider man mug`, `spider man jacket`, `spider man hoodies` and `spiderman t shirt`. 
- The total unique products for all search queries at the time of scraping was 2233. 
- We are considering this problem as a binary image classification problem.
- We need to predict whether the product image is a relavant one or non-relevant one.
- Our performance metrics will be f1-score and confusion matrix over accuracy since we need to know how well our model is going to identify the relevant products.

But do we have any labeled training data 🤔? 
Yes, we have partial labeled training data 😃💡.

- In the entire process of finding the relevant title using a string matching algorithm, we got around 1175 out of 2233 products as relevant ones, this is a form of weak supervision to label the data.

```python
start = time()
print("total no of products:", df.shape[0])
df["relevant_products_fuzz_match"] = df.title.apply(lambda x: string_match_with_fuzz(x))
print(f"time for processing: {time()-start} seconds", )
relevant_products = df[~df.relevant_products_fuzz_match.isnull()]
print("no of products identified:", relevant_products.shape[0])
relevant_products = relevant_products.drop_duplicates(subset=['image_url'])
print("no of relevant product images after removing duplicated image urls:", relevant_products.shape[0])
```
#### Output
```python
total no of products: 2233
time for processing: 0.20037364959716797 seconds
no of products identified: 1175
no of relevant product images after removing duplicated image urls: 1124
```

- Some of the products had the same image URLs, so after removing the duplicates we had around known 1124 relevant product images and we labeled them as class 1 or Target 1. 
- For class 2 or Target 0 data, we can make use of open-source datasets that are related to our domain i.e clothing, toys, and mugs. 
Target 0 images mean data that shouldn't have a spiderman image. 
- We need Images of T-shirts, hoodies, jackets, mugs, and toys that are not related to spiderman. We were able to source some clothing data from Kaggle. 

I am extremely thankful to the following kaggle dataset contributors.  
From their datasets, we were able to gather Target 0 images for our training data
1. https://www.kaggle.com/datasets/sunnykusawa/tshirts
2. https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl
3. https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

We ignored the available toys data in the 3rd dataset because there may be a chance of spiderman toys in that dataset and it will affect our model performance.

We can even use a pre-trained model in this process, but for now, I would like to do things from scratch for a better understanding of the problem and domain. 

So our final training data contains
1. 1124 relevant product images (Target 1)
2. Random Images (Target 0) were taken equally from the above 3 open source datasets 
3. In order to improve our model performance I have added around 48 toy images (Target 0 ) from our unidentified product images by manual work. 
<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218313268-1df7e7af-f998-4bc7-b922-a613308f72bf.png" width="600" height="300">
</p>
4. Similarly Extra 28 (Target 1) images, like spiderman logo images, spider-verse images from google Images, and some spiderman t-shirt images (from our unidentified product images by manual work) which were failed to identify by our model in the initial training stages. 
<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218313584-13ce3f6c-12bb-4398-976f-9ec4abde66ad.png" width="600" height="300">
</p>
<strong>NB</strong> : Since we only added 76 images manually (48 Target 0 + 28 Target 1). It wasn’t much of a time-consuming process. 

<img src="https://user-images.githubusercontent.com/78400305/218319617-413db23d-9765-4900-a671-862d73595f83.png" width="1000" height="200">

#### Model Training
##### Code: https://github.com/jithinanievarghese/image_classification_pytorch

Here we have trained a deep-learning CNN network using PyTorch. The training of the model was done using the following 
[Kaggle Notebook](https://www.kaggle.com/code/jithinanievarghese/image-classification-pytorch).  
Image Augmentation, Early Stopping and Dropout Layers were added to reduce overfitting.

#### Model Performance

Model with the best validation accuracy of **98.26%** was saved during the training. As we discussed earlier our model performance metric is really validated based on the f1 score and confusion matrix.
Even though our accuracy is **98.26%**, When we inspect the validation and training loss curve, we can see that the model is slightly overfitted.

<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218321163-82995623-754d-4171-a515-be93e6e1c782.png" width="900" height="400">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218321267-c8636b0c-dc0e-448a-ac14-ab79a10f8e22.png" width="600" height="400">
</p>

#### Inference
##### Code: [Inference notebook](https://github.com/jithinanievarghese/image_classification_pytorch/blob/main/inference.ipynb)  
To validate the perfomance of our model on unseen datasets, we gathered a subset of 28 images from unidentified products.
<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218323450-ded93a8e-0313-4569-90f2-9c18383eae65.png" width="900" height="400">
</p>

- The accuracy of our model on unseen data is only **76.92%** but we have a better f1 score of **80%**. 
- When we inspect the confusion matrix, out of 13 relevant products we predicted 12 products correctly, there was only one miss prediction.
- So model performs well in identifying the relevant products.
- Also, the model miss predicted 5 non-relevant products as Target 1 or relevant ones. But as we discussed earlier, our primary aim was to identify relevant products. 
- From that perspective, our model performed well with minimum manual labeling (76 images) and proper use of open-source datasets. 
- With further optimization techniques, we can achieve a better model which can perform well in the prediction of Target 0 images.


<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218323465-3376d5e1-c729-4f44-b72f-1b2e9f9d2433.png" width="650" height="350">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/78400305/218323478-a9c45b2a-bdcf-4d89-9579-a8f509098b1c.png" width="650" height="350">
</p>



##### Some of the optimization techniques are:

1. [Data-Centric Approach](https://analyticsindiamag.com/big-data-to-good-data-andrew-ng-urges-ml-community-to-be-more-data-centric-and-less-model-centric/) - We often try to improve the model by hyperparameter tuning and other regularization techniques, but this approach is more dedicated to improving the training data. In our case our model doesn’t perform well in predicting Target 0 images, so we have to add more labeled Target 0 images to our training data and re-train the model. The initial implementation of manually labeled 76 images gave us a far better performance. But the Data-Centric approach is expensive as we need to manually label the data or use any weak supervision techniques.
2. Using our existing model to label the unseen data and re-training our model with newly labeled data.
3. Fine Tuning pre-trained models like Inceptionv3, NASNetLarge, etc for our use case. But here the model size will be large. Large-size models can be a deployment concern for us, especially when considering the ROI.
4. Testing with more Image augmentation techniques other than [random zoom and brightness](https://github.com/jithinanievarghese/image_classification_pytorch/blob/36bf1836ebf23df804ef57c533830cf7b828d973/dataset.py#L25). Like we can augment the images and add it as additional data, then retrain the model.

#### Final Predictions
Detailed in [Inference notebook](https://github.com/jithinanievarghese/image_classification_pytorch/blob/main/inference.ipynb)

On the final predictions of model on unseen or unidentified product images, 317 products predicted as relevant and 198 as non-relevant
<p align="center">
  <img width="500" height="400" src="https://user-images.githubusercontent.com/78400305/218379926-9f52f902-70b9-4615-b1fe-10b461107f0a.png">
</p>

### Deployment, Scrape Flow and Conclusion

Since we need to find the relevant products during scraping, we need to deploy the model as an API (Currently deployment of the project is in progress.). Our model will be deployed as an API with input request of image path from our cloud platform or image url.  
We expect the probability of Target 1 and Target 0 in the response of that API call. 

- We will be using a combination of both NLP and image classification.
- At first we will try to identify the relevant products at the product title level by our string matching algorithm. 
- Major benefit is that we can reduce the image request for that identified product title, thereby reducing the cost in data gathering.
- Even in our case 1175 out of 2233 products were identifed from product title ie 52% of total data.
- Then we send requests to the product images for the products which were failed in string matching.
- And validate the images with our API call.
- Then we save the data

The expected flow of the Scrapy spider will be as follows:

<p align="center">
  <img width="300" height="700" src="https://user-images.githubusercontent.com/78400305/218374956-4306fdf8-25d0-494d-bc4c-bec66fa61f43.png">
</p>

### Resources

Data Gathering Spider(Scrapy): https://github.com/jithinanievarghese/flipkart_scraper_scrapy     
Data preprocessing:  [data preprocesing product search relevance](https://github.com/jithinanievarghese/flipkart_scraper_scrapy/blob/main/flipkart_scraper/flipkart_scraper/data%20preprocesing%20product%20search%20relevance.ipynb)  
Image Classification Code : https://github.com/jithinanievarghese/image_classification_pytorch   
Kaggle Notebook: https://www.kaggle.com/code/jithinanievarghese/image-classification-pytorch        
Inference Notebook: https://github.com/jithinanievarghese/image_classification_pytorch/blob/main/inference.ipynb

### References

1. [Practical Natural Language Processing](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/)
2. [Approaching (Almost) Any Machine Learning Problem](https://books.google.co.in/books/about/Approaching_Almost_Any_Machine_Learning.html?id=ZbgAEAAAQBAJ&source=kp_book_description&redir_esc=y)
3. https://scrapy.org/


