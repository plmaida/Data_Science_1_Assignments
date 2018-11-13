#!/usr/bin/env python
# -*- coding: utf-8 -*-

from newsapi import NewsApiClient
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import sqlite3
from dateutil.parser import parse
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import regexp_tokenize
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup 
from urllib.request import urlopen
import time
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output

# Function that pulls all the relevant news items going back a month
# Should not be used as it only pulls the first 1000 of the month total
# Instead use pull_one_day and use a loop to cover the full month
def pull_one_month(key):
    # Init
    newsapi = NewsApiClient(api_key=key)
    ps = 100

    # Init big list to hold results
    # Each entry is itself a list of: title, description, content, published at
    results = []

    # Run first request to determine size of response
    all_articles = newsapi.get_everything(q='crude AND oil',
                                        #from_param='2017-12-01',
                                        #to='2017-12-12',
                                        language='en',
                                        page_size=ps,
                                        sort_by='relevancy')

    # How many pages
    pages = int(all_articles['totalResults'] / ps + 1)
    print('Number of articles: ' + str(all_articles['totalResults']))
    print('Number of pages: ' + str(pages))

    # Crawl each page (can only crawl 10 pages x 100 articles with free subscription)
    for i in range(1, 11):
        # Request the page
        print('Processing page number ' + str(i))
        this_page = newsapi.get_everything(q='crude AND oil',
                                            #from_param='2017-12-01',
                                            #to='2017-12-12',
                                            language='en',
                                            page_size=ps,
                                            sort_by='relevancy',
                                            page=i)
        # Process each article into a row
        for j in range(0, ps):
            row = []
            for k in ['title', 'description', 'content', 'publishedAt']:
                row.append(this_page['articles'][j][k])
            results.append(row)
            
    # Confirm results were loaded correctly
    print(str(len(results)) + ' articles loaded')

    return results

# Function that pulls all the relevant news items since from_date
def pull_recent(from_date, key):
    # Init
    newsapi = NewsApiClient(api_key=key)
    ps = 100

    # Init big list to hold results
    # Each entry is itself a list of: title, description, content, published at
    results = []

    # Run first request to determine size of response
    recent_articles = newsapi.get_everything(q='crude AND oil',
                                        from_param=from_date,
                                        language='en',
                                        page_size=ps,
                                        #sort_by='relevancy'
                                        )
    
    # How many pages
    pages = int(recent_articles['totalResults'] / ps + 1)
    last_page_no = recent_articles['totalResults'] % ps
    print('Number of articles: ' + str(recent_articles['totalResults']))
    print('Number of pages: ' + str(pages))
    print('Number of articles on last page: ' + str(last_page_no))
    
    # Crawl each page (can only crawl 10 pages x 100 articles with free subscription)
    for i in range(1, pages+1):
        # Request the page
        print('Processing page number ' + str(i))
        this_page = newsapi.get_everything(q='crude AND oil',
                                            from_param=from_date,
                                            language='en',
                                            page_size=ps,
                                            sort_by='relevancy',
                                            page=i)
        # Process each article into a row
        x = ps
        
        if i == pages:
            x = last_page_no
        for j in range(0, x):
            row = []
            for k in ['title', 'description', 'content', 'publishedAt']:
                row.append(this_page['articles'][j][k])
            results.append(row)
            
    # Confirm results were loaded correctly
    print(str(len(results)) + ' articles loaded')
    
    return results

# Function that pulls all the relevant news items from a certain day
def pull_one_day(the_day, key):
    # Init
    newsapi = NewsApiClient(api_key=key)
    ps = 100

    # Init big list to hold results
    # Each entry is itself a list of: title, description, content, published at
    results = []

    # Run first request to determine size of response
    the_day_articles = newsapi.get_everything(q='crude AND oil',
                                             from_param=the_day,
                                             to=the_day,
                                             language='en',
                                             page_size=ps,
                                             sort_by='relevancy')

    # How many pages
    pages = int(the_day_articles['totalResults'] / ps + 1)
    last_page_no = the_day_articles['totalResults'] % ps
    print('Number of articles: ' + str(the_day_articles['totalResults']))
    print('Number of pages: ' + str(pages))
    print('Number of articles on last page: ' + str(last_page_no))

    # Crawl each page (can only crawl 10 pages x 100 articles with free subscription)
    for i in range(1, pages+1):
        # Request the page
        print('Processing page number ' + str(i))
        this_page = newsapi.get_everything(q='crude AND oil',
                                            from_param=the_day,
                                            to=the_day,
                                            language='en',
                                            page_size=ps,
                                            sort_by='relevancy',
                                            page=i)
        # Process each article into a row
        x = ps
        
        if i == pages:
            x = last_page_no
        for j in range(0, x):
            row = []
            for k in ['title', 'description', 'content', 'publishedAt']:
                row.append(this_page['articles'][j][k])
            results.append(row)
            
    # Confirm results were loaded correctly
    print(str(len(results)) + ' articles loaded')

    return results

# Function that creates the database
def make_db(name):
    conn = sqlite3.connect(name)
    c = conn.cursor()
    try:
        c.execute('''CREATE TABLE articles(
                title TEXT, description TEXT, 
                content TEXT, published_at TEXT)
                ''')
    except:
        print('Oops, the articles table already exists.')
    conn.commit()
    conn.close()

# Function that pulls historical prices and inserts them into the database
# It replaces the previous version of the table on successive calls
def pull_insert_historical_prices(db_name):
    # Pull up to date historical pricing
    wti = requests.get('https://www.quandl.com/api/v3/datasets/CME/CLZ2018.json?api_key=ZXzjihT3CgMKtPLPgTL9')
    json_data = json.loads(wti.text) 
    data_wti = json_data['dataset']['data']
    col = ['date', 'open', 'high', 'low', 'last', 'change', 'settle', 'volume', 'previous_day_open_interest']
    df = pd.DataFrame.from_records(data_wti, columns=col)

    # Insert the data into the database (replaces previous table, if it exists)
    conn = sqlite3.connect(db_name)
    df.to_sql("crude_index", conn, if_exists="replace")
    conn.commit()
    conn.close()

# Function that loads historical prices for graphing purposes
def load_historical_prices(db_name, no_of_rows):
    # Init
    date = []
    open_p = []
    high = []
    low = []
    last = []
    change = []
    
    # Load data
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    for row in c.execute('SELECT date, open, high, low, last, change FROM crude_index LIMIT ' + str(no_of_rows)):
        date.append(row[0])
        open_p.append(row[1])
        high.append(row[2])
        low.append(row[3])
        last.append(row[4])
        change.append(row[5])

    # Prepare results
    results = []
    results.append(date)    # 0
    results.append(open_p)  # 1
    results.append(high)    # 2
    results.append(low)     # 3
    results.append(last)    # 4
    results.append(change)  # 5

    return results

# Function that inserts new articles into the articles table
def insert_articles(articles, db_name):
    # Make a list of tuples
    data = []
    for a in articles:
        data.append(tuple(a))
    # Insert into database
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.executemany('INSERT INTO articles VALUES(?, ?, ?, ?)', data)
    conn.commit()
    conn.close()

# Function that returns all articles in the database
def load_all_articles(db_name):
    results = []
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    for row in c.execute('SELECT * FROM articles'):
        results.append(list(row))
    conn.commit()
    conn.close()
    return results

# Function that returns the articles in the database for a specific day
def load_day_articles(db_name, the_day):
    results = []
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query = "SELECT * FROM articles WHERE published_at LIKE '" + the_day[:10] + "%'"
    for row in c.execute(query):
        results.append(list(row))
    conn.commit()
    conn.close()
    return results

# Function that makes and inserts into a database every article for the last month
def pull_insert_one_month(db_name, key):
    #Make the database
    make_db(db_name)
    # Make a list of the days we are pulling as datetime objects
    days = []
    d = date.today() - relativedelta(months=1) + timedelta(days=1)
    while d.isoformat() < datetime.now().isoformat():
        days.append(d.isoformat())
        d = d + timedelta(days=1)
    print('***Loading '+ str(len(days)) + ' days...')

    # Loop through the days
    for i in days:
        print('***Loading day ' + str(i) + ' ...')
        # Pull articles for the day
        articles = pull_one_day(i, key)
        # Insert articles into the database
        insert_articles(articles, db_name)
        print('Done day ' + str(i) + '.')
    
    print('***Done pulling and inserting all articles for the monnth.')

# This function processes one article. It takes an article in the raw/db format
# Returns a list of the normalized, lematized words with stop words removed
def process_article(article):
    #Concatenate the article title, description and content
    article_data = ''
    for i in range (3):
        article_data = article_data + article[i]
    
    # Tokenize content
    words = word_tokenize(article_data)

    # Normalize the words
    normalized_words = [w.lower() for w in words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(w) for w in normalized_words]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in lem_words if not w in stop_words]

    # Make a list of bad words that need to be filtered out
    bad_words = [w for w in filtered_words if re.search(r'^\+|…$|\$?\d+(\.\d+)?%?|[][.,;"’():_—-]|[%&]', w)]
    # Add chars to ba words list (it appears by default in every single article)
    bad_words.extend(['char', 'percent', 'crude', 'oil', 'price', 'wti', 'ha', '``', "'s"])

    # Filter out bad words
    for bw in bad_words:
        c = filtered_words.count(bw)
        for i in range(c):
            filtered_words.remove(bw)

    # Return the list of filtered words
    return filtered_words

# Function that pulls live prices
def get_live_price():
    soup = BeautifulSoup(urlopen("https://markets.businessinsider.com/commodities/oil-price?type=wti"), 'html.parser')
    price = soup.find('span', {"class":"push-data "}).text
    return [datetime.now().isoformat(), price]

# Function that creates a matplotlib wordcloud figure
# It writes the cloud to the assets folder and returns a unique filename
def get_cloud(key):
    new_articles = pull_recent((datetime.now() - timedelta(minutes=15)).isoformat(), key)
    insert_articles(new_articles, 'project_a.db')
    # Load newly updated set of articles for the day
    today_articles = load_day_articles('project_a.db', date.today().isoformat())
    # Process the articles for the day into one big bag of words
    big_word_bag = []
    bad_article_count = 0
    for a in today_articles:
        try:
            big_word_bag.extend(process_article(a))
        except:
            bad_article_count += 1
    print('Skipped ' + str(bad_article_count) + ' articles as they had bad data.')
    # Display the wordcloud
    wordcloud = WordCloud(height=900, width=1600, scale=1, max_font_size=60, max_words=150, background_color="white")
    wordcloud.generate(' '.join(big_word_bag))
    filename = 'cloud' + datetime.now().isoformat() + '.png'
    wordcloud.to_file(os.getcwd() + '/assets/' + filename)

    return filename

