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

#%%selected US speeches from 2018

df_raw = speeches_data[speeches_data["country"] == "united states"]
df_raw = df_raw[df_raw["date"]>"2012-01-01"]
df_raw.sort_values("date",inplace = True)
df_raw.reset_index(drop = True, inplace = True)
df_raw.index.rename('speech_no', inplace=True) 
# group by country, time window
#%% keywords extraction
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
vectorizer = CountVectorizer(ngram_range= (2,2), stop_words="english")
# n-gram越小，越概括
df_speeches_kw = df_raw[["country","text","date"]]
df_speeches_kw["result"] = df_raw['text'].apply(lambda x : kw_model.extract_keywords(x, vectorizer=vectorizer,top_n=10,use_mmr = True,diversity = 0.5))
df_speeches_kw["result2"] = df_raw['text'].apply(lambda x : kw_model.extract_keywords(x, vectorizer=vectorizer,top_n=10,use_mmr = True,diversity = 0.5))

#%%

df1 = pd.DataFrame(df_speeches_kw["result2"].to_list(),columns =[f"k{i}" for i in range(10)])
df1.index = df_speeches_kw.index
df2 = pd.concat([df1["k0"],df1["k1"],df1["k2"],df1["k3"],df1["k4"],df1["k5"],df1["k6"],df1["k7"],df1["k8"],df1["k9"]]).to_frame()
df2.sort_index(inplace = True)
df3 = pd.DataFrame(df2[0].to_list(),columns =["keyword","similarity"])
df3["speech_no"] = df2.index 

#remove sim < 0.3
df4 = df3[df3["similarity"] >= 0.3]
df4.sort_values("speech_no",inplace = True)
df4["keyword_no"] = range(df4.shape[0]) #set keyword_index


#coun, to remove noise (optional)
#df_keyword_count = df_speeches_kw.groupby("keyword")["speech_no"].count()

#dedup and clustering (to find mapping relationship: keyword>>>topic)
li_keywords = set(df4['keyword'].to_list())
df_labeled_keywords = kmeans_cluster(li_corpus = li_keywords, n_clusters=20)
df_topic_kmeans = central_words(df_labeled_keywords) #to save 
test = df_topic_kmeans.T

#%% topic labeling (date, keyword, topic, country, sim)

df_mapping = df_topic_kmeans.iloc[:,0].to_frame()#label >>> topic(central words)
df_mapping.reset_index(inplace = True)
df_mapping.columns = ["label","topic"]

df_keywords_temp1 = df4.reset_index(drop = True).merge(df_speeches_kw[["date","country"]],on="speech_no")
df_keywords_temp2 = df_keywords_temp1.merge(df_labeled_keywords[["label","keyword"]], on = "keyword")
df_keywords = df_keywords_temp2.merge(df_mapping, on = "label")

#%%topic trend 
#time decay



#%% save results
#1. US speeches with speech_no
df_speeches_kw.to_csv(os.path.join(work_dir, "OUTPUT/central_bank_speech/all_speeches_with_keywords_v0.2.csv"))
df_topic_kmeans.T.to_csv(os.path.join(work_dir, "OUTPUT/central_bank_speech/keywords_groupping_n2_v0.2.csv"))
df_keywords.to_csv(os.path.join(work_dir, "OUTPUT/central_bank_speech/keywords_time_series_n2_v0.2.csv"))






