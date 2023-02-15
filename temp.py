# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Libraries for vectorization
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
#Libraries for embedding
from sentence_transformers import SentenceTransformer


#test
def func():
    print("yeah!")


# 1.1 vectorization
def CountVectorizer_func(doc,ngram_range = (1,1)):

    count = CountVectorizer(ngram_range= ngram_range, stop_words="english").fit([doc])
    candidates = count.get_feature_names_out()
    
    return candidates


def embedding_word_func(word):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(word)
    
    return embedding


