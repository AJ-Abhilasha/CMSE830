import streamlit as st
import seaborn as sns
import pandas as pd

col1, col2 = st.columns([1, 2])

col1.markdown(" # Songs app!")
col2.markdown(" Brief intro to the app")

df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

option = st.selectbox('select column name for showing the displot',('danceability_%', 'valence_%', 'energy_%',
       'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'))
st.write('You selected:', option)
plt2=sns.displot(df, x=option, hue="mode", multiple="stack")
st.pyplot(plt2.fig)

