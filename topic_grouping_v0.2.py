#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:51:57 2023

@author: yjy
"""
#%% import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer


from nlp_lib import kmeans_cluster,central_words

#%%1.load data
# dir
work_dir = os.getcwd()
input_path = os.path.join(work_dir, "INPUT/central_bank_speech/all_speeches.csv")
speeches_data = pd.read_csv(input_path)
speeches_data["date"] = pd.to_datetime(speeches_data["date"],format="%d/%m/%Y")
speeches_data.sort_values("date",inplace = True)
speeches_data.reset_index(drop = True, inplace = True)
speeches_data.index.rename('speech_no', inplace=True) 

# selected latest 20 row for test
df_raw = speeches_data.tail(10)

# group by country, time window
#%% keywords extraction
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
vectorizer = CountVectorizer(ngram_range= (1,2), stop_words="english")

df_keywords = df_raw[["text"]]
df_keywords["result"] = df_raw['text'].apply(lambda x : kw_model.extract_keywords(x, vectorizer=vectorizer,top_n=10,use_mmr = True,diversity = 0.5))

# n-gram越小，越概括

#%% keywords grouping

df1 = pd.DataFrame(df_keywords["result"].to_list(),columns =[f"k{i}" for i in range(10)])
df1.index = df_keywords.index
df2 = pd.concat([df1["k0"],df1["k1"],df1["k2"],df1["k3"],df1["k4"],df1["k5"],df1["k6"],df1["k7"],df1["k8"],df1["k9"]]).to_frame()
df2.sort_index(inplace = True)
df3 = pd.DataFrame(df2[0].to_list(),columns =["keyword","similarity"])
df3["speech_no"] = df2.index
df3.sort_values(["speech_no"],inplace = True)
df3.index.rename('keyword_no', inplace=True)  #to_save

#remove sim < 0.3
df4 = df3[df3["similarity"] >= 0.3]

#coun, to remove noise (optional)
#df_keyword_count = df4.groupby("keyword")["speech_no"].count()

#dedup and clustering (to find mapping relationship: keyword>>>topic)
li_keywords = set(df4['keyword'].to_list())
df_labeled_keywords = kmeans_cluster(li_corpus = li_keywords, n_clusters=10)
df_topic_kmeans = central_words(df_labeled_keywords) #to_save

#%% topic labeling and topic trend (date, keyword, topic, country, sim)







