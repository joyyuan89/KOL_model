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

st.markdown("1.introduction")
st.markdown("2.keyword search")
st.markdown("3.topic trend")
st.markdown("4.topic treemap")
