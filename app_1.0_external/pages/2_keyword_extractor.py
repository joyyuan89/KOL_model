# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
    layout="wide",
)

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

#%% select a speech

# Create a select box for the country column
selected_country = st.sidebar.selectbox("Select country", df["country"].unique(), index = 7)
# Filter the data based on the selected country
filtered_df = df[df["country"] ==selected_country]

# Create a select box for the author column, only showing dates from the selected country
selected_author = st.sidebar.selectbox("Select author", filtered_df["author"].unique())
# Filter the data based on the selected country and author
filtered_df = filtered_df[filtered_df["author"] == selected_author]


# Create a select box for the date column, only showing dates from the selected country and author 
selected_date = st.sidebar.selectbox("Select date", filtered_df.sort_values("date",ascending=False)["date"].unique())
# Filter the data based on the selected date
filtered_df = filtered_df[filtered_df["date"] == selected_date]

# Get full text, title and keyword for selected speech
doc = filtered_df["text"].values[0]
title = filtered_df["title"].values[0]


#%% display the speech


# st.image("logo.png", width=400)
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

st.markdown("### 🟠 1.Select a speech from the central bank speeches collection")

st.markdown(f"## :blue[_Title: {title}_]")

st.markdown(" #### 📌 View full text >>> ")

with st.expander("📔 -full text", expanded=False):

    st.markdown(doc)
    
   
#%% key word extractor

st.markdown("")
st.markdown(" ### 🟠 2.Extract key words ")
st.markdown("Please select the parameters for keyword extraction. After you have selected your parameters, click on the :blue[**_Get me the result_**] button to view the results. ")

with st.form(key="my_form"):


    ce, c1, ce, c2, ce, c3,ce = st.columns([0.07, 1, 0.07, 1, 0.07,1,0.07])
    with c1:
        
        ModelType = st.radio(
            "Choose your model",
            ["all-MiniLM-L6-v2","distilbert-base-nli-mean-tokens"],
            help="At present, you can choose between 2 pre_trained models to embed your text. More to come!",
        )
    
        if ModelType == "all-MiniLM-L6-v2":
            # kw_model = KeyBERT(model=roberta)
    
            @st.cache_resource()
            def load_model():
                return KeyBERT(model="all-MiniLM-L6-v2")
    
            kw_model = load_model()
    
        else:
            @st.cache_resource()
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")
    
            kw_model = load_model()
    
    with c2:

        top_N = st.slider(
            "Number of keywords",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.
    
    *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
    
    To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
                # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )
    
        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.
    
    *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
    
    To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )
        
    with c3:

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )
    
        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )
    
        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
    Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.
    
    """,
        )

        

    submit_button = st.form_submit_button(label="✨ Get me the result!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

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

st.markdown(" ### 🟠 2.Check results ")

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

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)


