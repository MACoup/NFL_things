# For The Win

The purpose of this project is to formalize a regimen for selecting players for a Daily Fantasy Football team.



Players earn points for their performance in each game. Here is a link to the scoring rules for Draft Kings fantasy football:

https://www.draftkings.com/help/nfl

# Motivation

The purpose of this project is to formalize a regimen for selecting players for a Daily Fantasy Football team. This repo is aimed at giving the user intelligent means to analyze and predict fantasy football performance measures in order to maximize win percentage.

# Tools

-   PostgreSQL
-   SQLAlchemy
-   Pandas
-   Numpy
-   Matplotlib
-   Sci-kit Learn

# How to

This repo requires the use of nfldb, which can be found here: https://github.com/BurntSushi/nfldb
Included in my code are SQL queries from this database which are read to .csv files using pandas and SQLAlchemy.

I have scraped the web for Draft Kings salary data and saved these as .csv. The use of .csv files is for ease of load in to the data analysis tools chosen by the user.

# Draft Kings

Because of the limited time period of this project, I have decided to focus on Draft Kings salary and contests. Inside the Draft Kings/Data folder is various scraped data. The folder with Draft Kings contest data is from a Primetime slate contest that I enter every week. There is also a folder containing all historical salary data for all NFL players. Draft Kings is relatively new, and as such only has salary data since 2014. There is also some other data scraped from the popular daily fantasy site rotogrinders. Here you can get projections and salary for players.

# NFLDB queries

This folder contains all my code for querying the nfldb, saving it to .csv, loading into a pandas data frame, and doing necessary cleaning and feature engineering. I found it best to load each position into a separate data frame, for ease of position specific features. It is easy to combine the dataframes together based on the slate of games you want and/or positions needed.

# EDA
In an effort to explore the feature space and discover which features were the most beneficial to which metrics. I used Recursive Feature Elimination to determine which features were best for each position.

# Models
The clustering models were implemented for an effort to classify players who might be outliers. The elliptic envelope is just one outlier detection algorithm I have used so far. Final_DF is a class to make it easy to create the dataframes you need. GBoost provides a way for classifying a player's points per $1000, with threshold at 3.5. At the same time, I am using a regressor to predict the exact value of points per $1000.
