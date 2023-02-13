import streamlit as st
import pandas as pd

df = pd.read_csv("OUTPUT/all_speeches_BERT.csv")

st.set_page_config(layout="wide")

# Create the keyword highlight function
def highlight_keywords(text, keywords):
    for word in keywords:
        text = text.replace(word, "<mark>" + word + "</mark>")
    return text


# Create a select box for the country column
selected_country = st.selectbox("Select country", df["country"].unique())
# Filter the data based on the selected country
filtered_df = df[df["country"] ==selected_country]

# Create a select box for the author column, only showing dates from the selected country
selected_author = st.selectbox("Select author", filtered_df["author"].unique())
# Filter the data based on the selected country and author
filtered_df = filtered_df[filtered_df["author"] == selected_author]


# Create a select box for the date column, only showing dates from the selected country and author 
selected_date = st.selectbox("Select date", filtered_df["date"].unique())
# Filter the data based on the selected date
filtered_df = filtered_df[filtered_df["date"] == selected_date]

# Create a select box for gram_of_keyphase
selected_gram_of_keyphase = st.selectbox("Select the gram of keyphase",[1,2,3])

# Get full text, title and keyword for selected speech
text = filtered_df["text"].values[0]
title = filtered_df["title"].values[0]
keywords = filtered_df["keywords"+"_"+str(selected_gram_of_keyphase)].values[0]


# Apply the highlight function to the full text
#text = highlight_keywords(text, keywords)

# Show the filtered data in a table
st.header(title)
st.subheader(keywords)
st.markdown(text)
