# import section
import pandas as pd
import numpy as np
import ast

# reading source files
df_movies = pd.read_csv(r'movie_recommendation\Sources\tmdb_5000_movies.csv')
df_credits = pd.read_csv(r'movie_recommendation\Sources\tmdb_5000_credits.csv')

# # printing first five records:
# print(df_credits.head(5))
# print(df_movies.head(5))

# merging the two dataset
df = df_movies.merge(df_credits,on='title')  # new

# # printing first five records:
# print(df.head(5))
# print(df.shape)
# print(df['original_language'].value_counts()) # new
# print(df.info()) # new
# print(df[['movie_id','title', 'overview', 'genres', 'keywords', 'cast', 'crew']])

# selecting required columns
df = df[['movie_id','title', 'overview', 'genres', 'keywords', 'cast', 'crew']] # new

# # checking for null values
# print(df.isnull().sum()) # new

# droping movies with null values
df.dropna(inplace=True) # new 

# # checking for null values
# print(df.duplicated().sum()) # new


# taking names of the genres out of the dictionary.
'''  [{"id": 28, "name": "Action"}, {"id": 12, "nam... '''
def convert (obj):
    res = []
    for i in ast.literal_eval(obj): #new
        res.append(i['name'])
    return res

df.genres = df.genres.apply(convert)

# taking keywords similar as genres
df.keywords = df.keywords.apply(convert)

# taking first 3 cast from the cast column
def convert3(obj):
    counter = 0
    res = []
    for i in ast.literal_eval(obj):
        if counter < 3:
            res.append(i['name'])
            counter += 1
        else:
            break
    return res
df.cast = df.cast.apply(convert3)

# getting details of director from the crew details
def get_director(obj):
    res = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            res.append(i['name'])
    return res

df.crew = df.crew.apply(get_director)

# converting the overview from string to list
df.overview =  df.overview.apply(lambda x: x.split()) # new

# removing space from the one name
df.genres = df.genres.apply(lambda x: [i.replace(" ", "") for i in x])
df.keywords = df.keywords.apply(lambda x: [i.replace(" ", "") for i in x])
df.crew = df.crew.apply(lambda x: [i.replace(" ", "") for i in x])
df.cast = df.cast.apply(lambda x: [i.replace(" ", "") for i in x])

# creating tag column with concatenation of other columns:
df['tags'] = df.overview + df.genres + df.keywords + df.cast + df.crew

# removing original columns used to create tag:
df = df [['movie_id', 'title', 'tags']]

# converting list of tags into space seperated string and converting it to lowecas
df.tags = df.tags.apply(lambda x: " ".join(x))
df.tags = df.tags.apply(lambda x: x.lower())

# applying stemming on the tag
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    res = []
    for i in text.split():
        res.append(ps.stem(i))
    return " ".join(res)

df.tags = df.tags.apply(stem)

# creating machine leaning model
from sklearn.feature_extraction.text import CountVectorizer # new
cv = CountVectorizer(max_features=5000, stop_words='english') # new
vectors = cv.fit_transform(df['tags']).toarray() # new

# calculating cosine distance between vectors
from sklearn.metrics.pairwise import cosine_similarity # new
similarity = cosine_similarity(vectors) # new

# getting top 5 movies for recommendation
def recommend(movie):
    movie_index = df[df['title']== movie].index[0] # new
    dists = similarity[movie_index]
    top_5 = sorted(list(enumerate(dists)), reverse=True, key=lambda x:x[1])[1:6]
    for i in top_5:
        print(df.iloc[i[0]].title)
    
recommend('Batman Begins')


