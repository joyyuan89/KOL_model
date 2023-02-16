#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:31:18 2023

@author: jiayue.yuan
"""

#%%Libraries 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


from nlp_lib import keyBERT,embedding_word_func,kmeans_cluster,central_words,pca_func,dbscan_cluster

#%%1.load data
# dir
work_dir = os.getcwd()
input_path = os.path.join(work_dir, "INPUT/central_bank_speech/all_speeches.csv")
speeches_data = pd.read_csv(input_path)
speeches_data["date"] = pd.to_datetime(speeches_data["date"],format="%d/%m/%Y")

# selected latest 20 row for test
df_raw = speeches_data.set_index("date").tail(10)

# group by country, time window

#%% 2.keyword extraction

li_keywords = [] # list of (keywords list for each speech)
li_keyword_embeddings = []

for doc in df_raw["text"]:
    keywords1, keyword_embeddings1 = keyBERT(doc=doc,ngram_range = (1,1), mss = False)
    
    
    li_keywords.append(keywords1)
    li_keyword_embeddings.append(keyword_embeddings1)
    
#%%
vectorizer = KeyphraseCountVectorizer()
kw_model.extract_keywords(doc, vectorizer=vectorizer)
    
#%%
# update df_speeches
df_speeches = df_raw.copy()
df_speeches['keywords'] = li_keywords
df_speeches['keyword_embeddings'] = li_keyword_embeddings

#%% 3. data prepossing for clustering: flatten list and dedup

#length of li_all keywords and li_all_embeddings doesn't match?
# set is unordered data structure!!!!
li_all_keywords = set(list(np.concatenate(li_keywords).flat))
#li_all_embeddings = [word for sublist in li_keywords for word in sublist]
#li_all_embeddings = [embedding for sublist in li_keyword_embeddings for embedding in sublist]

#%% 3.1 k-means clustering
df_labeled_keywords = kmeans_cluster(li_corpus = li_all_keywords, n_clusters=10)
#df_labeled_keywords = kmeans_cluster(li_corpus = li_all_keywords,li_embeddings=li_all_embeddings, n_clusters=10)
df_topic_kmeans = central_words(df_labeled_keywords, n_words=20)
            
#%% 3.2DBscan
#DBscan
li_all_embeddings_2d = pca_func([embedding_word_func(word) for word in li_all_keywords])                              
df_labeled_keywords_db = dbscan_cluster(li_corpus = li_all_keywords, li_embeddings=li_all_embeddings_2d, eps = 0.2, min_samples = 5)

df_topic_dbscan = central_words(df_labeled_keywords_db, n_words=5)

# plot the clusters
df_plot = pd.DataFrame(li_all_embeddings_2d)
pic3 = plt.scatter(df_plot[[0]], df_plot[[1]], c=df_labeled_keywords_db["label"], cmap='rainbow')
    
             
#%% 4.save results
df_speeches.to_csv(os.path.join(work_dir, "OUTPUT/keyword_embeddings.csv"))
df_topic_kmeans.to_csv(os.path.join(work_dir, "OUTPUT/topic_list_kmeans.csv"))
df_topic_dbscan.to_csv(os.path.join(work_dir, "OUTPUT/topic_list_dbscan.csv"))
