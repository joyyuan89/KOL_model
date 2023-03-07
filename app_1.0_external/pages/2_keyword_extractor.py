# -*- coding: utf-8 -*-
"""
Created on Mar 7

@author: jiayue.yuan
"""

import streamlit as st
import pandas as pd
from pandas import DataFrame
import io
from keybert import KeyBERT
import seaborn as sns
# For download buttons
#from functionforDownloadButtons import download_button

# google cloud data storage
from google.oauth2 import service_account
from google.cloud import storage


st.set_page_config(
    page_title="Keyword Extractor",
    page_icon="🎈",
    layout="wide",
)

#%% parameters
ModelType = "all-MiniLM-L6-v2"
StopWords = "english"
mmr = True
Diversity = 0.5

#%% load data 

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
@st.cache_data
def read_csv_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    csv_file_content = bucket.blob(file_path).download_as_string()
    return csv_file_content

bucket_name = "kol_model"
file_path = "INPUT/central_bank_speech/all_speeches.csv"
csv_data = read_csv_file(bucket_name, file_path)

df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))

#%% load model
@st.cache_resource()
def load_model():
    return KeyBERT(model="all-MiniLM-L6-v2")

kw_model = load_model()

#%% page intro

st.title("🔑 BERT Keyword Extractor")
st.header("")

with st.expander("ℹ️ - About this page", expanded=True):
    st.write(
        """     
- Select a speech from the central bank speeches collection by specifying country, speaker and date in the sidebar.
- Customize your keyword extraction options based on your needs.
- View the results to see the extracted keywords and their relevance to the speech.
- Refine your analysis by adjusting the options and extracting keywords again.
	    """
    )

    st.markdown("")

st.markdown("")

#%% select a speech
st.markdown("### 🟠 1.Select a speech from the central bank speeches collection")

ce, c1, ce, c2, ce, c3,ce = st.columns([0.07, 1, 0.07, 1, 0.07,1,0.07])

with c1:
    #select box for countries
    selected_country = st.selectbox("Select country", df["country"].unique(), index = 7)
    filtered_df = df[df["country"] ==selected_country]

with c2:
    # select box for the author column, only showing dates from the selected country
    selected_author = st.selectbox("Select author", filtered_df["author"].unique())
    # Filter the data based on the selected country and author
    filtered_df = filtered_df[filtered_df["author"] == selected_author]

with c3:
    # select box for the date column, only showing dates from the selected country and author 
    selected_date = st.selectbox("Select date", filtered_df.sort_values("date",ascending=False)["date"].unique())
    # Filter the data based on the selected date
    filtered_df = filtered_df[filtered_df["date"] == selected_date]

# Get full text, title and keyword for selected speech
doc = filtered_df["text"].values[0]
title = filtered_df["title"].values[0]

st.markdown(f"## :blue[_Title: {title}_]")

st.markdown(" #### 📌 View full text >>> ")

with st.expander("📔 -full text", expanded=False):

    st.markdown(doc)   
   
#%% key word extractor

st.markdown("")
st.markdown(" ### 🟠 2.Extract keywords/keyphrases ")
st.markdown(""" Selected parameters, and click the :blue[**_Get me the result_**] button to view the results.  
            :blue[**Keyphrase ngram range**] sets the length of the resulting keywords/keyphrases.
            """)
            
with st.form(key="my_form"):

    ce, c1, ce, c2, ce, c3,ce = st.columns([0.07, 1, 0.07, 1, 0.07,1,0.07])

    
    with c1:

        top_N = st.slider(
            "Number of Keyword/Keyphrase",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
    
    with c2:
        
        min_Ngrams = st.number_input(
            "Min Ngram",
            min_value=1,
            max_value=4,
            help="The minimum value for the keyphrase_ngram_range."
        )
    
    with c3:
    
        max_Ngrams = st.number_input(
            "Max Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="The maximum value for the keyphrase_ngram_range.")

    submit_button = st.form_submit_button(label="✨ Get me the result!")

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
)

#%% check results

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 4, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)


