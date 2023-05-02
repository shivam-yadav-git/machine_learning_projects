# import section
import pandas as pd
import numpy as np

# reading source files
df_movies = pd.read_csv(r'movie_recommendation\Sources\tmdb_5000_movies.csv')
df_credits = pd.read_csv(r'movie_recommendation\Sources\tmdb_5000_credits.csv')

# printing first five records:
print(df_credits.head(5))
print(df_movies.head(5))

# merging the two dataset
df = df_movies.merge(df_credits,on='title')  # new

# printing first five records:
print(df.head(5))
print(df.shape)
print(df['original_language'].value_counts()) # new
print(df.info()) # new
print(df[['movie_id','title', 'overview', 'genres', 'keywords', 'cast', 'crew']])

# selecting required columns
df = df[['movie_id','title', 'overview', 'genres', 'keywords', 'cast', 'crew']] # new

# checking for null values
print(df.isnull().sum()) # new

# droping movies with null values
df.dropna(inplace=True) # new 

# checking for null values
print(df.duplicated().sum()) # new