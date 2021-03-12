import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from parse import *

pd.options.display.max_colwidth = 500
np.set_printoptions(precision=2)

#movies

base_url = "https://www.imdb.com"
movie_tags = ['/title/tt0097757/', '/title/tt0103639/', '/title/tt0101414/', '/title/tt0110357/','/title/tt0120762/']
movie_links = [base_url + tag + 'reviews' for tag in movie_tags]
print("There are a total of " + str(len(movie_links)) + " movie user reviews")
print("Displaying 10 user reviews links")
movie_links[:10]

# get a list of soup objects
movie_soups = [getSoup(link) for link in movie_links]

# get all movie review links
# movie_review_list = [getReviews(movie_soup) for movie_soup in movie_soups]
movie_review_list = []
for movie_soup in movie_soups :
    movie_review_list.append(getReviews(movie_soup))
movie_review_list = list(itertools.chain(*movie_review_list))
print(len(movie_review_list))

print("There are a total of " + str(len(movie_review_list)) + " individual movie reviews")
print("Displaying 10 reviews")
movie_review_list[:10]

# get review text from the review link
review_texts = [getReviewText(url) for url in movie_review_list]

# get movie name from the review link
movie_titles = [getMovieTitle(url) for url in movie_review_list]

# label each review with negative or positive
review_sentiment = np.array(['negative', 'positive'] * (len(movie_review_list)//2))

# construct a dataframe
df = pd.DataFrame({'movie': movie_titles, 'user_review_permalink': movie_review_list,
             'user_review': review_texts})

df.head()

# save the dataframe to a csv file.
df.to_csv('userReviews.csv', index=False)

# pickle the dataframe
df.to_pickle('userReviews.pkl')

# to validate
#temp = pd.read_csv('userReviews.csv')
#temp = pd.read_pickle('userReviews.pkl')


userReviewDF = df['user_review']    #Data frame of only user reviews 
count = CountVectorizer()
bag = count.fit_transform(userReviewDF)
#print(count.vocabulary_)    #prints array with corresponding frequency
#print(bag.toarray())        #prints out # of times word at index appears

#weighing importance of words based on frequency
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

vectorizer = TfidfVectorizer(lowercase=True, max_features=100)
X = vectorizer.fit_transform(userReviewDF)
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print(features)
#print(features)







