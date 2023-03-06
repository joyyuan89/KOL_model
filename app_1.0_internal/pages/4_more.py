#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:14:22 2023

@author: jiayue.yuan
"""

import streamlit as st
import pandas as pd

from google.oauth2 import service_account
from google.cloud import storage

st.set_page_config(
    page_title="more_info",
    page_icon="🎈",
    layout="wide",
)



# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data()
def read_excel_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    excel_file_content = bucket.blob(file_path).download_as_bytes()
    
    return excel_file_content

bucket_name = "kol_model"
file_path = "INPUT/reference_tables/weight.xlsx"
file_content = read_excel_file(bucket_name, file_path)

reference_table_country = pd.read_excel(file_content, sheet_name="country")
reference_table_topic_list = pd.read_excel(file_content, sheet_name="topic list")


with st.expander("ℹ️ - About this page", expanded=True):

    st.write(
        """     
        Thank you for using our app! We hope that you found the insights provided by our app valuable and informative.\n
        We are constantly working to refine and improve our methodology, and welcome feedback and suggestions from you!  
        Please find more details of our model below:
	    """
    )

    st.markdown("")


st.markdown("#### 📄 Topic List")
st.markdown("This is the topic list we used in :blue[**Topic Trend Tracker**]")

st.dataframe(reference_table_topic_list)

st.markdown("#### 📄 Country Weights")
st.markdown("We applied weighting approach to account for differences in central bank policies and priorities across countries, and ensures that the trend indices reflect the broader trends in the global financial landscape.")

st.dataframe(reference_table_country)

st.markdown("#### 📄 Time decay function")

st.markdown("To be added...")
