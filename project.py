import streamlit as st
import seaborn as sns
import pandas as pd

col1, col2 = st.columns([1, 2])

col1.markdown(" # Songs app!")
col2.markdown(" Brief intro to the app")

df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

# Removing incorrect data from file 
target_value = 'BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3'
df = df[df['streams'] != target_value]
df['streams'] = df['streams'].astype(int)

# Creating a new column with standardized values for #streams
scaler = StandardScaler()
df['s_streams'] = scaler.fit_transform(df[['streams']])



fig = px.scatter(df,
    x="valence_%",
    y="energy_%",
    size="streams",
    color="mode",
    hover_name="track_name",
    log_x=True,
    size_max=60,
)
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)
