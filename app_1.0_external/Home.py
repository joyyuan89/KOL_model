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
#%%main page


st.markdown("# Home Page 🎈")

#1.introduction
st.markdown("### 1.Introduction")
st.markdown("""
            Welcome to our cutting-edge Natural Language Processing (NLP) project app!  \n
            Our app is designed to scan and analyze text-based market information using the power of transformed-based language models. \n
            We have leveraged the power of the :blue[**BERT transformer**] method to analyze the :blue[**speeches given by central bank officials**]. 
            By extracting keywords from the speeches and tracing the popularity of meaningful topics over the years, we can analyze how central banks' priorities and concerns have evolved over time. \n           
            This insight is invaluable for policymakers, economists, and researchers, who need to stay up-to-date with the latest developments in the financial markets.          
            """)
#2. guide for keyword extractor

st.markdown("### Part1. Topic Trend Tracker")
st.markdown("""This part involves tracking keyword trends over time, providing a valuable insight into how market sentiment and conditions change over time.  
            Also,a keyword heat map is generated, which provides a visual representation of the central bank speech's most significant keywords.  
            This heat map is particularly useful in identifying trends and patterns in the data, allowing for easier interpretation and analysis.
            """)
            
            
st.markdown("### Part2. Keyword Extractor")
st.markdown("This part retrieves one specific speech of central bank speeches collection, and analyzes the speech using keyword analysis to extract important pieces of information.")


     
            
#%% sidebar
st.markdown(""" > ✨About the authors:  
            Jake Wen  
            Jiayue Yuan""")





