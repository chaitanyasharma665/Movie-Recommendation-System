# Import necessary libraries
import numpy as np
import pandas as pd
import ast  # For parsing strings containing lists or dictionaries


# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# Merge movies and credits datasets on the 'title' column
movies = movies.merge(credits, on='title')


# Retain only the required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# Drop rows with missing values
movies.dropna(inplace=True)


# Define a function to convert JSON-like strings into Python lists
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])  # Extract the 'name' field from each dictionary
    return L 

    

# Apply the conversion function to the 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)



# Converts a string representation of a list of dictionaries into a list of 'name' values from the first 3 dictionaries.
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter <3 :
            L.append(i['name'])
        counter+=1
    return L 



# Apply the 'convert' function to the 'cast' column of the 'movies' DataFrame
movies['cast'] = movies['cast'].apply(convert3)



# Function to extract names of directors from a given text
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# Apply the fetch_director function to extract the director from the 'crew' column
movies['crew'] = movies['crew'].apply(fetch_director)


# Tokenize the 'overview' column by splitting each string into a list of words
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# Remove spaces from genre names in the 'genres' column
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])


# Remove spaces from keywords in the 'keywords' column
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])


# Remove spaces from cast members' names in the 'cast' column
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])


# Remove spaces from crew members' names in the 'crew' column
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# Combine relevant columns (overview, genre, keywords, cast, and crew) into a single 'tags' column for easier processing
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# Combine all elements in the 'tags' list into a single space-separated string for each row in the DataFrame.

new_df=movies[['movie_id','title','tags']]


# Convert the list of tags in each row of the 'tags' column into a single string separated by spaces
new_df.loc[:,'tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# Convert all text in the 'tags' column to lowercase for uniformity
new_df.loc[:,'tags'] = new_df['tags'].apply(lambda x: x.lower())


# Import the CountVectorizer class from the sklearn.feature_extraction.text module
from sklearn.feature_extraction.text import CountVectorizer


# Create a CountVectorizer object with specific parameters
cv = CountVectorizer(max_features=5000, stop_words='english')



# Convert the 'tags' column into a numerical representation using CountVectorizer
# Step 1: Learn the vocabulary of unique words in the 'tags' column and count their occurrences (fit_transform)
# Step 2: Convert the resulting sparse matrix into a dense NumPy array for easier processing (toarray)
vector = cv.fit_transform(new_df['tags']).toarray()


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



# Defining the stem function
def stem(text):
    
    # Create an empty list to store the stemmed words
    y = []
    
    # Loop through each word in the input text after splitting it by whitespace
    for i in text.split():
        
        # Apply stemming to the word 'i' using the Porter Stemmer (ps.stem)
        y.append(ps.stem(i))
        
    # Return the list of stemmed words as a single space-separated string
    return " ".join(y)


# Apply stemming to the 'tags' column to reduce words to their root form
new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)


# Import cosine_similarity to compute the similarity between movie vectors
from sklearn.metrics.pairwise import cosine_similarity


# Calculate cosine similarity for all movie vectors
# This creates a similarity matrix where each entry [i][j] indicates how similar movie i is to movie j
similarity = cosine_similarity(vector)


# Function to recommend movies based on similarity
def recommend(movie):
    
    # Find the index of the movie in the DataFrame using the title
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    
    # Get the similarity scores for the given movie with all other movies
    distances = similarity[movie_index]
    
    # Sort movies based on their similarity scores in descending order
    # Enumerate helps keep track of the indices, which are needed for fetching movie titles
    # We skip the first movie ([1:6]) because it will always be the input movie itself (highest similarity)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    
    # Print the titles of the top 5 most similar movies
    for i in movies_list:
        print(new_df.iloc[i[0]]['title'])
        

# Input prompt to allow the user to enter a movie name
m = input('Enter the movie: ')

# Call the recommend function with the input movie
recommend(m)
