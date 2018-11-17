# oil_dashboard
WTI Crude Pricing Dashboard

The dashboard consists of two python files and an empty “assets” folder. The runme.py file runs the dash server and does all the background work. The oil_data_lib.py file contains the functions we wrote for web scraping, data wrangling and database transactions.

The assets folder is used to store the word cloud photos displayed by in the dashboard. It is empty before the program is run, but will slowly fill out with a new image for every update. The folder does not clean itself, so an eye should be kept on its growing size. In a future version of this code, we plan on automatic this as part of the cleanup (we were forced to create new files each time in order to get around a stubborn browser caching issue that would not results in the image updating). The program will create its own database file in the folder it is run from.

The system takes a few minutes to initialize, given the data loading and model training required upfront. Once the terminal output stops displaying new lines, the system is running. Dash produces a web server that can be accessed by simply going to http://127.0.0.1:8050/ in any browser. More details are available in the terminal output of our program.

It is worth noting that many of the packages described above, and which are imported in our code, might require installation before the Dash can run properly. All of those can be installed quickly with simple pip install commands.

Ideally, the program should be restarted every day. The reason for this is that it allows the classifier to be retrained on a new dataset that includes the previous day.

This was worked on with a group member. 
References: quandl.com/data/ODA/POILWTI_USD-WTI-Crude-Oil-Price
            https://markets.businessinsider.com/commodities/oil-price?type=wti
            
