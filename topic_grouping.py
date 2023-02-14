#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:40:35 2023

@author: jiayue.yuan
"""


#%% 1. Doc & Word Embedding

#Libraries for vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
#Libraries for embedding
from sentence_transformers import SentenceTransformer


# 1.1 vectorization
def CountVectorizer_func(doc,ngram_range = (1,1)):

    count = CountVectorizer(ngram_range= ngram_range, stop_words="english").fit([doc])
    candidates = count.get_feature_names_out()
    
    return candidates


def KeyphraseCountVectorizer_func(doc):
    
    #pos_pattern = '<J.*>*<N.*>+',
    count = KeyphraseCountVectorizer(doc,stop_words="english").fit([doc])
    candidates = count.get_feature_names_out()
    
    return candidates


# 1.2 Embedding
# pretrained models: 'all-MiniLM-L6-v2', 'distilbert-base-nli-mean-tokens'

def embedding_func(doc, candidates, model_name = 'all-MiniLM-L6-v2'):
    
    model = SentenceTransformer(model_name)
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    
    return doc_embedding, candidate_embeddings

def embedding_word_func(word):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(word)
    
    return embedding

def decode_word_func(vector):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    word = model.decode(vector)
    
    return word
    

#%% 2. Keywords Extraction (cosine_similarity)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools

# for n_gram
def extract_keywords(candidates, doc_embedding, candidate_embeddings, top_n = 5):
    
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    keyword_embeddings = [candidate_embeddings[index] for index in distances.argsort()[0][-top_n:]]
    
    return keywords, keyword_embeddings
    

def max_sum_sim( candidates, doc_embedding, candidate_embeddings, top_n =5, nr_candidates = 10):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    words_embeddings = [candidate_embeddings[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim
            
    keywords = [words_vals[idx] for idx in candidate]
    keyword_embeddings = [words_embeddings[idx] for idx in candidate]

    return keywords, keyword_embeddings

#%% 3. customized keyBERT

def keyBERT(doc,ngram_range, mss = False):
    
    candidates = CountVectorizer_func(doc,ngram_range = ngram_range)
    doc_embedding, candidate_embeddings = embedding_func(doc, candidates, model_name = 'all-MiniLM-L6-v2')
        
    if mss is False:
        keywords, keyword_embeddings = extract_keywords(candidates, doc_embedding, candidate_embeddings)
    
    if mss is True:
        keywords, keyword_embeddings = max_sum_sim( candidates, doc_embedding, candidate_embeddings, top_n =5, nr_candidates = 10)

    return keywords, keyword_embeddings

#%% 4. word clustering

from sklearn.cluster import KMeans

def kmeans_cluster(li_all_keywords, n_clusters=10):
    
    df_keywords = pd.DataFrame(li_all_keywords,columns = ["keyword"])
    df_keyword_embeddings = pd.DataFrame(df_keywords["keyword"].apply(embedding_word_func).to_list())
                 
    #kmeans
    clustering_model = KMeans(n_clusters)
    clustering_model.fit(df_keyword_embeddings)
    
    # find cluster center
    center_vectors = clustering_model.cluster_centers_
    df_sim = pd.DataFrame(cosine_similarity(df_keyword_embeddings, center_vectors))
    
    # update label of df_keywords 
    labels = clustering_model.fit_predict(df_keyword_embeddings)
    df_labeled_keywords = pd.concat([df_keywords,pd.DataFrame(labels,columns = ["label"]),df_sim],axis = 1)
    
    return df_labeled_keywords

def central_words(df_labeled_keywords,n_clusters = 10, n_words=10):
    
    """
    df_labeled_keywords dataframe:
    columns: keyword label 0  1  2  3  4.... (similarity with each center)
    """
    
    dic_topic_clusters = {}
    for cluster in range(n_clusters):
        df_cluster = df_labeled_keywords[df_labeled_keywords["label"] == cluster]
        # top n words in each cluster
        li_central_words = df_cluster.sort_values(by = cluster, ascending=False)["keyword"].to_list()[0:n_words]
        dic_topic_clusters[cluster] = li_central_words
    
    df_topic = pd.DataFrame(dic_topic_clusters)
        
    return df_topic

"""
main body

"""
#%%1.load data

import pandas as pd
import os

# dir
work_dir = os.getcwd()
input_path = os.path.join(work_dir, "INPUT/central_bank_speech/all_speeches.csv")
speeches_data = pd.read_csv(input_path)
speeches_data["date"] = pd.to_datetime(speeches_data["date"],format="%d/%m/%Y")

# selected latest 20 row for test
df_raw = speeches_data.set_index("date").tail(200)

# group by country, time window

#%% 2.keyword extraction

li_keywords = [] # list of (keywords list for each speech)
li_keyword_embeddings = []

for doc in df_raw["text"]:
    keywords1, keyword_embeddings1 = keyBERT(doc=doc,ngram_range = (1,1), mss = False)
    #keywords2, keyword_embeddings2 = keyBERT(doc=doc,ngram_range = (2,2), mss = True)
    
    li_keywords.append(keywords1)
    li_keyword_embeddings.append(keyword_embeddings1)
    
# update df_speeches
df_speeches = df_raw.copy()
df_speeches['keywords'] = li_keywords
df_speeches['keyword_embeddings'] = li_keyword_embeddings

#%% 3.k-means

#flatten list and dedup
li_all_keywords = li_all_keywords = set(list(np.concatenate(li_keywords).flat))

df_labeled_keywords = kmeans_cluster(li_all_keywords, n_clusters=10)
df_topic = central_words(df_labeled_keywords,n_clusters = 10, n_words=10)
              
#%% 4.plot topic trend
# key_word_search.py
               
#%% 5.plot topic trend
df_speeches.to_csv(os.path.join(work_dir, "OUTPUT/keyword_embeddings.csv"))
df_topic.to_csv(os.path.join(work_dir, "OUTPUT/topic_list.csv"))
                   













    
        

        

        
        


 





    
    
    



