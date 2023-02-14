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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

#4.1 PCA

def pca_func(X,n_components=2): # X can be a list of arrays

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_2d = pca.transform(X)
    
    return X_2d

#4.2 kmeans

def kmeans_cluster(li_corpus, li_embeddings = None, n_clusters=10):
    #li_embeddings: list of arrays
    #li_corpus: list of strs
    
    df_corpus = pd.DataFrame(li_corpus,columns = ["keyword"])
    
    if li_embeddings is None: # generate embeddings
        df_embeddings = pd.DataFrame(df_corpus["keyword"].apply(embedding_word_func).to_list())
    
    else:
        df_embeddings = pd.DataFrame(li_embeddings)   
        
                
    #kmeans
    clustering_model = KMeans(n_clusters)
    clustering_model.fit(df_embeddings)
    
    # find cluster center
    center_vectors = clustering_model.cluster_centers_
    df_sim = pd.DataFrame(cosine_similarity(df_embeddings, center_vectors))
    
    # update label of df_keywords 
    labels = clustering_model.fit_predict(df_embeddings)
    df_labeled_corpus = pd.concat([df_corpus,pd.DataFrame(labels,columns = ["label"]),df_sim],axis = 1)
    
    return df_labeled_corpus

#4.3 DBscan

def dbscan_cluster(li_corpus, li_embeddings = None, eps = 0.3, min_samples = 10):
    
    df_corpus = pd.DataFrame(li_corpus,columns = ["keyword"])
    
    if li_embeddings is None: # generate embeddings
        df_embeddings = pd.DataFrame(df_corpus["keyword"].apply(embedding_word_func).to_list())
    
    else:
        df_embeddings = pd.DataFrame(li_embeddings) 
        
    # normalization ?
    df_embeddings =  StandardScaler().fit_transform(df_embeddings)
        
    # DBscan
    clustering_model = DBSCAN(eps = eps, min_samples = min_samples)
    clustering_model.fit(df_embeddings)
    labels = clustering_model.fit_predict(df_embeddings)
    
    # find cluster center
    # Calculate the mean center of each cluster
    unique_labels = set(labels)

    centers = {}
    for label in unique_labels:
        if label != -1:
            cluster = df_embeddings[clustering_model.labels_ == label]
            center = np.mean(cluster, axis=0)
            centers[label] = center
        
    center_vectors = pd.DataFrame(centers).T
    df_sim = pd.DataFrame(cosine_similarity(df_embeddings, center_vectors))
    
    # update label of corpu and similarity to centers
    df_labeled_corpus = pd.concat([df_corpus,pd.DataFrame(labels,columns = ["label"]),df_sim],axis = 1)
    
    return df_labeled_corpus 


#4.4 find words near cluster centers

def central_words(df_labeled_keywords, n_words=10):
    
    """
    df_labeled_keywords dataframe:
    columns: keyword label 0  1  2  3  4.... (similarity with each center)
    """
    
    labels = df_labeled_keywords["label"].unique()
    
    dic_topic_clusters = {}
    for label in labels:
        if label != -1: # -1 refers to noise in DBscan result
            df_cluster = df_labeled_keywords[df_labeled_keywords["label"] == label]
            # top n words in each cluster
            li_central_words = df_cluster.sort_values(by = label, ascending=False)["keyword"].to_list()[0:n_words]
            dic_topic_clusters[label] = li_central_words
    
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
df_raw = speeches_data.set_index("date").tail(500)

# group by country, time window

#%% 2.keyword extraction

li_keywords = [] # list of (keywords list for each speech)
li_keyword_embeddings = []

for doc in df_raw["text"]:
    keywords1, keyword_embeddings1 = keyBERT(doc=doc,ngram_range = (1,1), mss = False)
    #keywords2, keyword_embeddings2 = keyBERT(doc=doc,ngram_range = (1,2), mss = True)
    
    li_keywords.append(keywords1)
    li_keyword_embeddings.append(keyword_embeddings1)
    
# update df_speeches
df_speeches = df_raw.copy()
df_speeches['keywords'] = li_keywords
df_speeches['keyword_embeddings'] = li_keyword_embeddings

#%% 3.clustering

#flatten list and dedup
#length of li_all keywords and li_all_embeddings doesn't match?
# set is unordered data structure!!!!
li_all_keywords = set(list(np.concatenate(li_keywords).flat))
#li_all_embeddings = [word for sublist in li_keywords for word in sublist]
#li_all_embeddings = [embedding for sublist in li_keyword_embeddings for embedding in sublist]


#3.1 k-means
df_labeled_keywords = kmeans_cluster(li_corpus = li_all_keywords, n_clusters=10)
#df_labeled_keywords = kmeans_cluster(li_corpus = li_all_keywords,li_embeddings=li_all_embeddings, n_clusters=10)
df_topic_kmeans = central_words(df_labeled_keywords, n_words=5)

#3.2 DBscan
li_all_embeddings_2d = pca_func([embedding_word_func(word) for word in li_all_keywords])                              
df_labeled_keywords_db = dbscan_cluster(li_corpus = li_all_keywords, 
                                      li_embeddings=li_all_embeddings_2d,
                                      eps = 0.2, 
                                      min_samples = 5)

df_topic_dbscan = central_words(df_labeled_keywords_db, n_words=10)
              
#%% 4.visulization of DBscan
# key_word_search.py
import matplotlib.pyplot as plt

# plot the clusters
df_plot = pd.DataFrame(li_all_embeddings_2d)
pic3 = plt.scatter(df_plot[[0]], df_plot[[1]], c=df_labeled_keywords_db["label"], cmap='rainbow')
    
             
#%% 5.save results
df_speeches.to_csv(os.path.join(work_dir, "OUTPUT/keyword_embeddings.csv"))
df_topic_kmeans.to_csv(os.path.join(work_dir, "OUTPUT/topic_list_kmeans.csv"))
df_topic_dbscan.to_csv(os.path.join(work_dir, "OUTPUT/topic_list_dbscan.csv"))



                   













    
        

        

        
        


 





    
    
    




