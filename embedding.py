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
import spacy
nlp = spacy.load("en_core_web_sm")

# dir
work_dir = os.getcwd()

# load data
input_path = "/Users/jakewen/Desktop/Github/KOL_model/INPUT/central_bank_speech/all_speeches.csv"

speeches_data = pd.read_csv(input_path)
speeches_data['date'] = pd.to_datetime(speeches_data['date'], format="%d/%m/%y")
speeches_data.set_index('date', inplace=True)
speeches_data.dropna(inplace=True)

# sampling for test
# speeches_data = speeches_data.sample(10)

#%% Shorten text

def shorten_text(text, tag_value):
    # load and break text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    shortened_sentences = sentences[:int(len(sentences)*tag_value)]
    shortened_sentences = ' '.join(shortened_sentences)
    
    return shortened_sentences

# full or shortened text
tag = "full"

if tag == "full":
    tag_value = 1
elif tag == "shortened":
    tag_value = 0.5
else:
    raise Exception("Unknown tag value")
    
speeches_data["text"] = speeches_data["text"].progress_apply(lambda x: shorten_text(x, tag_value))
speeches_data.to_excel("speeches_data_shortened.xlsx")

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
speeches_data.to_csv(embedder_name+"_embedding_"+tag+".csv")
