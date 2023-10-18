import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import plotly.express as px

st.markdown(""" <style> .font_title {
font-size:40px ; font-family: 'courier'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:30px ; font-family: 'times'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:20px ; font-family: 'sans'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font_title">Tops Songs on Spotify</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Data and variables", "Plots", "Predictions"])

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
    
    dataset = st.checkbox('Show summary of songs dataset')
    if dataset == True:
        st.table(df.describe())
        st.markdown('<p class="font_subtext">Table 1: Spotify\'s top songs </p>', unsafe_allow_html=True)

    # Creating a pie chart of keys in the songs
    st.markdown('<p class="font_subheader">Categorical attributes in the dataset</p>', unsafe_allow_html=True)

    key1_name = df['key'].value_counts().index
    key1_value = df['key'].value_counts()

    key2_name = df['mode'].value_counts().index
    key2_value = df['mode'].value_counts()

    fig3, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(key1_value, labels=key1_name, autopct='%1.1f%%', colors= sns.color_palette("Set3", 11))
    axes[0].set_title("Keys in songs")
    axes[1].pie(key2_value, labels=key2_name, autopct='%1.1f%%', colors= sns.color_palette("pastel", 2))
    axes[1].set_title("Modes in songs")
    for ax in axes:
        ax.axis('equal')
    st.pyplot(fig3) 

    chart = alt.Chart(df).mark_bar().encode(
        alt.X('bpm:Q', bin=True),
        alt.Y('count():Q', title='Frequency'),
        alt.Color('mode:N')
    ).facet(
        column=alt.Column('mode:O', header=alt.Header(title=None))
    ).configure_axisY(
        grid=True
    )
    st.markdown('<p class="font_subheader">Plot of beats per minute</p>', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)

with tab3:
    st.markdown("Note that the size of the bubbles in the following plots correspond to the number of times a particular song was streamed")
    #Creating a scatterplot of user selected song attribute
    options = st.multiselect('Select 2 song attributes to plot', ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'], default=['valence_%', 'energy_%'], max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

    fig1 = px.scatter(df,
        y=options[0],
        x=options[1],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=25,
    )
    st.markdown('<p class="font_subheader">Plot of the two selected variables</p>', unsafe_allow_html=True)
    st.plotly_chart(fig1, theme=None, use_container_width=True)

    #Creating a scatterplot of user selected charts
    chart_options = st.multiselect('Select 2 song attributes to plot', ['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts'], default=['in_spotify_charts', 'in_apple_charts'], max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

    fig1 = px.scatter(df,
        y=chart_options[0],
        x=chart_options[1],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=25,
    )
    st.markdown('<p class="font_subheader">Plot of the two selected variables</p>', unsafe_allow_html=True)
    st.plotly_chart(fig1, theme=None, use_container_width=True)
