#%% Setup
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 11:12:50 2023

@author: jakewen
"""

# libraries
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.csv"

speeches_data = pd.read_csv(input_path)
speeches_data['date'] = pd.to_datetime(speeches_data['date'], format="%d/%m/%Y")
speeches_data.set_index('date', inplace=True)

# sampling for test
# speeches_data = speeches_data.sample(10)

#%% Embedding model
from sentence_transformers import SentenceTransformer

# embedder_name = 'multi-qa-mpnet-base-dot-v1' # heavy weight semantic search
# embedder_name = 'multi-qa-MiniLM-L6-cos-v1' # light weight semantic search
# embedder_name = 'all-MiniLM-L6-v2' # light weight all-rounder
embedder_name = 'all-mpnet-base-v2' # heavy weight all-rounder

def embedding(text):
    embedder = SentenceTransformer(embedder_name)
    doc_embeddings = embedder.encode(text)

    return doc_embeddings

#%% Main body
import time
start_time = time.time()
print("program running")

speeches_data["text_embedding"] = speeches_data["text"].progress_apply(lambda x: embedding(x))

print("program completed")
print("--- %s sec ---" % (time.time() - start_time))

#%% Download
speeches_data.to_excel(embedder_name+"_embedding.xlsx")
