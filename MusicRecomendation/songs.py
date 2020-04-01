import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation
import sys

song= sys.argv[1]
triplets_file = 'Data.txt'
songs_metadata_file = 'song_data.csv'
song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
song_df_2 =  pandas.read_csv(songs_metadata_file)
#merge
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 
song_df.head()
song_df = song_df.head(10000)

song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
train_data.head(5)
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
users = song_df['user_id'].unique()
user_id = users[3]
pm.recommend(user_id)
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')
is_model.get_similar_items([song])


