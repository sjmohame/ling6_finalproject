##############################
#  Module: imdbUtils.py
#  Author: Shravan Kuchkula
#  Date: 07/13/2019
##############################

import requests
import random
import re
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

porter = PorterStemmer()


def getSoup(url):
    """
    Utility function which takes a url and returns a Soup object.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    return soup


def minMax(a):
    '''Returns the index of negative and positive review.'''

    # get the index of least rated user review
    minpos = a.index(min(a))

    # get the index of highest rated user review
    maxpos = a.index(max(a))

    return minpos, maxpos


def getReviews(soup):
    '''Originally returned highest and lowest review by rating, edited to return 25 random reviews.'''

    # get a list of user ratings
    user_review_ratings = [tag.previous_element for tag in
                           soup.find_all('span', attrs={'class': 'point-scale'})]

    # find the index of negative and positive review
    n_index, p_index = minMax(list(map(int, user_review_ratings)))

    # get the review tags
    user_review_list = soup.find_all('a', attrs={'class': 'title'})

    # get the negative and positive review tags
    n_review_tag = user_review_list[n_index]
    p_review_tag = user_review_list[p_index]

    # return the negative and positive review link
    n_review_link = "https://www.imdb.com" + n_review_tag['href']
    p_review_link = "https://www.imdb.com" + p_review_tag['href']

    # return random list of reviews
    ran_list = random.sample(user_review_list, 25)

    for i in range(25):
        curr = ran_list[i]
        ran_list[i] = "https://www.imdb.com" + curr['href']
    return ran_list


def getReviewText(review_url):
    '''Returns the user review text given the review url.'''

    # get the review_url's soup
    soup = getSoup(review_url)

    # find div tags with class text show-more__control
    tag = soup.find('div', attrs={'class': 'text show-more__control'})

    return tag.getText()


def getMovieTitle(review_url):
    '''Returns the movie title from the review url.'''

    # get the review_url's soup
    soup = getSoup(review_url)

    # find h1 tag
    tag = soup.find('h1')

    return list(tag.children)[1].getText()


def getNounChunks(user_review):

    # create the doc object
    doc = nlp(user_review)

    # get a list of noun_chunks
    noun_chunks = list(doc.noun_chunks)

    # convert noun_chunks from span objects to strings, otherwise it won't pickle
    noun_chunks_strlist = [chunk.text for chunk in noun_chunks]

    return noun_chunks_strlist

##################
    # Source: "Performing Sentiment Analysis on Movie Reviews"
    # Author: "Bryan Tan"
    # Link: "https://towardsdatascience.com/imdb-reviews-or-8143fe57c825"
##################

def preprocessor(text):
    text =re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def tokenizer(text):
    return text.split()

def tokenizer_stemmer(text):
    return[porter.stem(word) for word in text.split()]
