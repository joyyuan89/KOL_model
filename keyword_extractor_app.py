#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:30:07 2023

@author: jiayue.yuan
"""

import streamlit as st
import numpy as np
from pandas import DataFrame
import os
import json
import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button

from nlp_lib import keyBERT



st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
    layout="wide",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("🔑 BERT Keyword Extractor")
    st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for extract keywors in central_bank_speech
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )

    st.markdown("")

st.markdown("")
#st.markdown("## **📌 Paste document **")
st.markdown("## **📌 Select Speech **")

with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your embedding model",
            #["DistilBERT (Default)", "Flair"],
            ["all-MiniLM-L6-v2","distilbert-base-nli-mean-tokens"],
            help="At present, you can choose between 2 pre_trained models to embed your text. More to come!",
        )

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=5,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            value = 1,
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


        use_MSS = st.checkbox(
            "Use MSS",
            value=True,
            help="You can use Max Sum Similarity (MSS) to diversify the results.",
        )
        
        nr_candidates = st.slider(
            "Keyword diversity (MMR only)",
            value=10,
            min_value=1,
            max_value=30,
            step=1,
            help="""must be greater than topN.
            
Note that the *Keyword diversity* slider only works if the *MSS* checkbox is ticked.
""",
        )


    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the data!")

    if use_MSS:
        mss = True
    else:
        mss = False

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()
    
if top_N > nr_candidates:
    st.warning("top_N can't be greater than nr_candidates")
    st.stop()

keywords, keyword_embeddings = keyBERT(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mss=mss,
    nr_candidates = nr_candidates,
    topN=top_N,
    model_name = ModelType
)

st.markdown("## **🎈 Check & download results **")

st.header("")

cs, c1, cLast = st.columns([2, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")

st.header("")

#df = (
#    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
#    .sort_values(by="Relevancy", ascending=False)
#    .reset_index(drop=True)
#)

df = DataFrame(keywords)

"""
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
"""