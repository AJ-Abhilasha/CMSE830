import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import plotly.express as px

#col1, col2 = st.columns([3, 1])

#col1.markdown(" # Top Songs on Spotify!")
st.markdown(""" <style> .font_title {
font-size:40px ; font-family: 'courier'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:20px ; font-family: 'sans'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

##########

st.markdown('<p class="font_title">Tops Songs on Spotify</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Introduction", "Data and variables"])

with tab1:
    st.markdown('<p class="font_text">Ever wonder what makes a song a hit? If so, you are at the right place! Here we\'ll explore the data and try to answer the following questions. </p>', unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - What musical attributes (e.g., tempo, danceability, energy) are common among top-streamed songs?</p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Are there any trends or patterns in terms of the mode of the song?</p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Do certain artists have a consistent presence among top-streamed songs? </p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Are there any temporal or seasonal trends in song popularity? </p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Are there any trends in a song\'s performance with respect to the number of artists involved? </p>
    """, unsafe_allow_html=True)


with tab2:
    df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

    # Removing incorrect data from file 
    target_value = 'BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3'
    df = df[df['streams'] != target_value]
    df['streams'] = df['streams'].astype(int)

    # Creating a new column with standardized values for #streams
    scaler = StandardScaler()
    df['s_streams'] = scaler.fit_transform(df[['streams']])
    
    dataset = st.checkbox('Show songs dataset')
    if dataset == True:
        st.table(df.describe())
        st.markdown('<p class="font_subtext">Table 1: Spotify\'s top songs </p>', unsafe_allow_html=True)


    options = st.multiselect('Select 2 variables to plot', ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'], default=None, max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

    fig1 = px.scatter(df,
        y=options[0],
        x=options[1],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=15,
    )
    st.plotly_chart(fig1, theme=None, use_container_width=True)
