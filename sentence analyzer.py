#%% Setup
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:04:42 2023

@author: jakewen
"""

import spacy
nlp = spacy.load("en_core_web_sm")

# libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.csv"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

speeches_data = pd.read_csv(input_path)
speeches_data['date'] = pd.to_datetime(speeches_data['date'], format="%d/%m/%y")
speeches_data.set_index('date', inplace=True)

# sampling for test
speeches_data = speeches_data.sample(1)

text = speeches_data['text'][0]

doc = nlp(text)
sentences = pd.DataFrame([sent.text.strip() for sent in doc.sents])
sentences.columns = ['text']
sentences.iloc[-1] = speeches_data['text']

print(sentences)

#%% Embedding model
from sentence_transformers import SentenceTransformer

# embedder_name = 'multi-qa-mpnet-base-dot-v1' # heavy weight semantic search
# embedder_name = 'multi-qa-MiniLM-L6-cos-v1' # light weight semantic search
embedder_name = 'all-MiniLM-L6-v2' # light weight all-rounder
# embedder_name = 'all-mpnet-base-v2' # heavy weight all-rounder

def embedding(text):
    embedder = SentenceTransformer(embedder_name)
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

#%% Embedding
import time
start_time = time.time()
print("program running")

sentences["text_embedding"] = sentences["text"].progress_apply(lambda x: embedding(x))

print("program completed")
print("--- %s sec ---" % (time.time() - start_time))

sentences_copy = sentences.copy()

#%% Cosine similarity
from numpy.linalg import norm

def cosine_similarity_function(vec_1, vec_2):
    value = np.dot(vec_1, vec_2.T)/(norm(vec_1)*norm(vec_2))
    return value

#%% Calculate similarity value

# scaling factor
power = 4
rolling_window = 6

sentences = sentences_copy

sentences["similarity_value"] = sentences["text_embedding"].apply(
    lambda x: cosine_similarity_function(
    x, sentences.iloc[-1, sentences.columns.get_loc("text_embedding")]))
sentences = sentences[:-1]

sentences = sentences.copy()
sentences.loc[:,"similarity_value_pwr"] = sentences.loc[:,"similarity_value"].pow(power)
sentences.loc[:,"similarity_value_adj"] = sentences.loc[:,"similarity_value_pwr"].rolling(
    window=rolling_window,
    min_periods=1,
    center=True,
    ).mean()

# normalize to 0-1 range
sentences.loc[:,"similarity_value_scaled"] = (sentences["similarity_value_adj"]-sentences["similarity_value_adj"].min())/(sentences["similarity_value_adj"].max()-sentences["similarity_value_adj"].min())

#%% Plot
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,6)

x = sentences.index
y = sentences["similarity_value_scaled"]

plt.xlabel("Sentence position")
plt.ylabel("similarity_value")
plt.title("Sentence analysis")

plt.plot(x, y)
plt.show()

#%% Clustering

'''
# K means clustering
from sklearn.cluster import KMeans

kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=30,
    max_iter=300,
    )

# Umap
import umap

reducer = umap.UMAP(
    n_components=2,
    )

reduced = reducer.fit_transform(sentences["text_embedding"].to_list())

#%% Visualization
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

fig = px.scatter(
    reduced, 
    x=0, y=1, 
    title='2D_plot', 
    height=700,
    width=700,
    )

fig.show()
'''