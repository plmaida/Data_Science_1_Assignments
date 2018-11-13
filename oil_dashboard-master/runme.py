#!/usr/bin/env python

import oil_data_lib as odl
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

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from numpy import array, unique

db_name = 'project_a.db'
# Load key for news API
news_key = '31713d1f33a5412a8b9ffcb2c663a661'

# Remove old instance of the database, if it exists
try:
    os.remove(db_name)
except:
    pass

# Clean up previous word cloud files, if they exist
# If there is time

# Init database for prices and news
odl.make_db(db_name)

# Pull and insert historical prices
odl.pull_insert_historical_prices(db_name)

# Insert previous month's news
odl.pull_insert_one_month(db_name, news_key)

# Some more init
time_series = []
price_day = []

print('Preparing classifier training dataset...')
# Make a classifier training dataset using the previous month's pricing data and news
# Load prices for the last 20 days (5 weekdays x 4)
price_data = odl.load_historical_prices(db_name, 20)
dates_change = [price_data[0], price_data[5]]
print(dates_change)
# Make labels
# absolute percent change >= 1% --> 1
# absolute percent change < 1% --> 0
for i in range(0, 20):
    if dates_change[1][i] >= 1:
        dates_change[1][i] = 1
    else:
        dates_change[1][i] = 0

dataset = []

# Cycle through the days in the price_change dataset
for i in range(0, 20):
    # Load articles for the day
    day_articles = odl.load_day_articles(db_name, dates_change[0][i])
    for j in range(0, len(day_articles)):
        # Process only articles that have complete information
        try:
            row = []
            row.append(odl.process_article(day_articles[j]))
            row.append(dates_change[1][i])
            dataset.append(row)
        except:
            pass

# Prepare data for training pipeline
pipeline_data = []
pipeline_target = []
for i in range(0, len(dataset)):
    pipeline_data.append(' '.join(dataset[i][0]))
    pipeline_target.append(dataset[i][1])

# Run the dataset through a pipeline combining
# a text feature extractor with a simple classifier

print("Defining pipeline parameters...")
# Define pipeline
pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())
])

# Uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (5,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

# Peform grid search for the optimal parameters
if __name__ == '__main__':
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(pipeline_data, pipeline_target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# We have now trained the best model
# We will run the model for the current day's new with every update

# THE DASHBOARD
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H1('WTI Crude Pricing Dashboard'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000*60*15, # in milliseconds
            n_intervals=0
        )
    ])
)

@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    # Get live prices
    [time_stamp, price] = odl.get_live_price()
    style = {'padding': '5px', 'fontSize': '16px'}
    
    # Get new word cloud
    fname = odl.get_cloud(news_key)
    
    # Check for news since the last update and run prediction model
    today_articles = odl.pull_one_day(date.today().isoformat(), news_key)
    # Process the data to be pipeline-ready
    predict_set = []
    for x in today_articles:
        # Append articles with complete information
        try:
            predict_set.append(' '.join(odl.process_article(x)))
        except:
            pass
    # Run model on data
    results = grid_search.predict(predict_set)
    uni, counts = unique(results, return_counts=True)
    try:
        ones = dict(zip(uni, counts))[1]
    except:
        ones = 0
    pred_probability = float(ones) / len(results)
    print('Predicted probability of +_1 percent change today is :')
    print(float(ones) / len(results))
    return [
        html.Span('Price: $' + price, style=style),
        html.Span('Time Stamp: ' + parse(time_stamp).strftime('%H:%M:%S'), style=style),
        html.Span('Predicted probability for 1% change in price today: ' + str(pred_probability*100) + '%', style=style),
        html.Img(src=app.get_asset_url(fname), style={'width': '900px'})
    ]

# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    # Get live prices
    [time_stamp, price] = odl.get_live_price()
    time_series.append(parse(time_stamp))
    price_day.append(price)
    # Get historical prices
    price_data = odl.load_historical_prices('project_a.db', 100)

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=3, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': price_data[0],
        'y': price_data[1],
        'name': 'Opening Price',
        'mode': 'lines+markers',
        'type': 'scatter',
        'line': {
            'dash': 'dash'
        }
    }, 1, 1)
    fig.append_trace({
        'x': price_data[0],
        'y': price_data[4],
        'text': price_data[0],
        'name': 'Closing Price',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': price_data[0],
        'y': price_data[3],
        'text': price_data[0],
        'name': 'Low',
        'mode': 'lines+markers',
        'type': 'scatter',
        'line': {
            'dash': 'dashdot'
        }
    }, 1, 1)
    fig.append_trace({
        'x': price_data[0],
        'y': price_data[2],
        'text': price_data[0],
        'name': 'High',
        'mode': 'lines+markers',
        'type': 'scatter',
        'line': {
            'dash': 'dot'
        }
    }, 1, 1)
    fig.append_trace({
        'x': price_data[0],
        'y': price_data[5],
        'text': price_data[0],
        'name': 'Change',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)
    fig.append_trace({
        'x': [w.strftime('%H:%M:%S') for w in time_series],
        'y': price_day,
        'text': [w.strftime('%H:%M:%S') for w in time_series],
        'name': 'Live Price',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 3, 1)
    


    return fig


if __name__ == '__main__':
    app.run_server(debug=True)