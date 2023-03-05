#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:55:07 2023

@author: jiayue.yuan
"""

import streamlit as st


st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': "mailto:joyyuan89@gmail.com",
        'Report a bug': "mailto:joyyuan89@gmail.com",
        'About': "contact author: Jiayue & Jake"
    }
)

#%% sidebar
st.sidebar.markdown("# ✨About the authors")

st.sidebar.write("Jake Wen")
st.sidebar.write("Jiayue Yuan")
#%%main page


st.markdown("# Home Page 🎈")

#1.introduction
st.markdown("### 1.Introduction")
st.markdown("""
              Welcome to our cutting-edge Natural Language Processing (NLP) project app! 
            Our app is designed to scan and analyze text-based market information using the power of transformed-based language models. 
            Our main objective is to distill information and generate an information hierarchy for analysis, which provides valuable insights to businesses and investors alike.\
              We have leveraged the power of the BERT transformer method to analyze the speeches given by central bank officials. 
            By extracting keywords from the speeches and tracing the popularity of meaningful topics over the years, we can analyze how central banks' priorities and concerns have evolved over time. 
            This insight is invaluable for policymakers, economists, and researchers, who need to stay up-to-date with the latest developments in the financial markets.\
              What sets our project apart is its user-friendliness. Anyone can customize the embedding model, adjust the searching parameters, and search for any topic they are interested in. 
            The app provides a customized experience for every user, allowing them to gain deep insights into the trends and patterns of the financial markets.\
              Overall, our NLP project app has the potential to revolutionize the way market information is analyzed. 
            It provides a valuable tool for anyone interested in the financial markets, from investors to business owners. 
            With our app, you can gain a deeper understanding of the market trends and make informed decisions based on data-driven insights.
            
            """)
#2. guide for keyword extractor
st.markdown("### Part1. Keyword Extractor")

st.markdown("This part retrieves one speech of central bank speeches collection, and analyzes the speech using keyword analysis to extract important pieces of information.")


st.markdown("### Part2. Topic Trend Tracker")
st.markdown("""The second part involves tracking keyword trends over time, providing a valuable insight into how market sentiment and conditions change over time. \
            Also,a keyword heat map is generated, which provides a visual representation of the central bank speech's most significant keywords. 
            This heat map is particularly useful in identifying trends and patterns in the data, allowing for easier interpretation and analysis.
            """)

st.markdown("### Part3. Topic Searcher")
st.markdown("This page enable users to search any topic they are interested in and view the topic trend.")


st.markdown("### Part4. More...")
st.markdown(""" Please note that our project is still under development, we welcome any feedback or suggestions. so we have included some details of our calculation process on the last page of the app.
            Your feedback will help us to refine and improve our algorithms. 
            """)
            
st.markdown("> generated by ChatGPT")




