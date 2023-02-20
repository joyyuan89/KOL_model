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
st.sidebar.markdown("# ✨About the app")

st.sidebar.write(
        """     
The app collects central bank speeches from 1900s to 2022. 
Ulitizing NLP tool of Bert, clustering algo and other ML techniques, 
the app can extract keywords from speech text, plot trends of market narratives.
	    """
    )

#%%main page


st.markdown("# Home Page 🎈")

#1.introduction
st.markdown("### 1.Introduction")
st.markdown("Market narrative is a good indicator for market trend......")

#2. guide for keyword extractor
st.markdown("### 2.Keyword extractor")
st.markdown(" First, select a speech you want to view. Then keywords extraction options like embedding model......")

#3.
st.markdown("### 3.Topic trend")

st.markdown("#### >> 2.1 Search topics and view trends")
st.markdown(" You can search a single topics or compare multiple topics' trends")

st.markdown("#### >> 2.2 topic popularity treemap")
st.markdown(" This treemap shows the most hot topic nowadays......")

