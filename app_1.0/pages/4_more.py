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

st.markdown("#### 📄 Topic List")

st.dataframe(reference_table_topic_list)

st.markdown("#### 📄 Country Weight")

st.dataframe(reference_table_country)
