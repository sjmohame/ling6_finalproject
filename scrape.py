##############################
#  Module: imdbUtils.py
#  Author: Shravan Kuchkula
#  Date: 07/13/2019
##############################

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

base_url = "https://www.imdb.com" #Picking movies from IMDB's database
#Disney movie list
movie_tags = ['/title/tt0097757/', '/title/tt0103639/', '/title/tt0101414/', '/title/tt0110357/','/title/tt0120762/']
#Ghibli movie list
movie_tags2 = ['/title/tt0096283/', '/title/tt0245429/', '/title/tt0347149/', '/title/tt0097814/', '/title/tt0095327/']
#For each movie in either of the lists, create a url for the review lists by adding the base url and the review tag
#Note, this program runs one or the other, so change the list name in the loop in order to check the other list
movie_links = [base_url + tag + 'reviews' for tag in movie_tags]
print("There are a total of " + str(len(movie_links)) + " movie user reviews")
print("Displaying 10 user reviews links")
movie_links[:10]

# get a list of soup objects
movie_soups = [getSoup(link) for link in movie_links]

# get all movie review links
# Use the getReviews function to generate a random list of 25 reviews for each movie
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

##################
    # Source: "Performing Sentiment Analysis on Movie Reviews"
    # Author: "Bryan Tan"
    # Link: "https://towardsdatascience.com/imdb-reviews-or-8143fe57c825"
##################

userReviewDF = df['user_review']    #Data frame of only user reviews 
count = CountVectorizer()           # Create a vector with words followed by their counts
bag = count.fit_transform(userReviewDF) 

vectorizer = TfidfVectorizer(lowercase=True, max_features=100) #Uses TDIDF to down-weight unnecessary words
                                                               #Converts then all to lowercase 
                                                               # for more accurate comparison
y = vectorizer.fit_transform(userReviewDF)                     
commonWords = vectorizer.get_feature_names()                      #Get the keys for each of the top 100 words
print(commonWords)                                                #Print them out as a list to the terminal







