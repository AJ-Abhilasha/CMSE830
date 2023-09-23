import plotly.express as px
import streamlit as st

st.write("""
# Iris Dataset
Exploring the relationship between different physical aspects of 
the flower Iris as a function of its species.
""")

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', 
    y='sepal_width', z='petal_length', color='species')

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)