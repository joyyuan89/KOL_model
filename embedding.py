#%% Setup
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:12:50 2023

@author: jakewen
"""

# libraries
import pandas as pd
import os

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.csv"

speeches_data = pd.read_csv(input_path)
speeches_data['date'] = pd.to_datetime(speeches_data['date'], format="%d/%m/%Y")
speeches_data.set_index('date', inplace=True)

#%% Embedding model
from sentence_transformers import SentenceTransformer

def embedding(text):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

#%% Main body
import time
start_time = time.time()
print("program running")

speeches_data["text_embedding"] = speeches_data["text"].apply(lambda x: embedding(x))

print("program completed")
print("--- %s min ---" % (time.time() - start_time))

#%% Post processing

date_range = pd.date_range(start=speeches_data.index.min(), end=speeches_data.index.max(),freq='D')
date_range = date_range.to_frame()
df_output = pd.merge(speeches_data, date_range, left_index=True, right_index=True, how='outer')
df_output.fillna(0,inplace=True)

#%% Download
df_output.to_csv("df_output.csv")