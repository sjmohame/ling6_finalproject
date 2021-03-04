import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import itertools
from parse import *

pd.options.display.max_colwidth = 500

#movies

base_url = "https://www.imdb.com"
movie_tags = ['/title/tt0097757/']
movie_links = [base_url + tag + 'reviews' for tag in movie_tags]
print("There are a total of " + str(len(movie_links)) + " movie user reviews")
print("Displaying 10 user reviews links")
movie_links[:10]

# get a list of soup objects
movie_soups = [getSoup(link) for link in movie_links]

# get all 500 movie review links
movie_review_list = [getReviews(movie_soup) for movie_soup in movie_soups]

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
             'user_review': review_texts, 'sentiment': review_sentiment})

df.head()

# save the dataframe to a csv file.
df.to_csv('userReviews.csv', index=False)

# pickle the dataframe
df.to_pickle('userReviews.pkl')

# to validate
#temp = pd.read_csv('userReviews.csv')
#temp = pd.read_pickle('userReviews.pkl')