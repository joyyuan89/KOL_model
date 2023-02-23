#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:14:22 2023

@author: jiayue.yuan
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="more_info",
    page_icon="🎈",
    layout="wide",
)


@st.cache_data
def load_data(local_path):
    df = pd.read_csv(local_path)
    return df


#1.2 reference data
dic_reference_data = {}
input_path_ref = "/Users/jiayue.yuan/Documents/Github/KOL_model/INPUT/reference_tables/weight.xlsx"
dic_reference_data["country_weight"] = pd.read_excel(input_path_ref, sheet_name="country")
dic_reference_data["topic_list"] = pd.read_excel(input_path_ref, sheet_name="topic list")

reference_table_country = dic_reference_data["country_weight"]
reference_table_topic_list = dic_reference_data["topic_list"]

st.markdown("#### 📄 Topic List")

st.dataframe(reference_table_topic_list)

st.markdown("#### 📄 Country Weight")

st.dataframe(reference_table_country)

