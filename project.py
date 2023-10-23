import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import streamlit as st
import plotly.express as px

st.markdown("""<style> .font_title {
font-size:40px; font-family:'courier'; color:black; text-align:center;} 
</style>""", unsafe_allow_html=True)

st.markdown("""<style> .font_subheader {
font-size:30px ; font-family: 'times'; color: black;text-align: center;} 
</style>""", unsafe_allow_html=True)

st.markdown("""<style> .font_text {
font-size:20px ; font-family: 'sans'; color: black;text-align: left;} 
</style>""", unsafe_allow_html=True)

st.markdown('<p class="font_title">The Symphony of Success</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Data and variables", "Plots",
                                        "Predictions", "Conclusions"])

with tab1:
    st.markdown("""<p class="font_text">Music is a universal and
    ever-present force in our lives. It\'s likely that you\'ve already
    encountered several songs today. While music has long captivated
    the attention of researchers across diversefields such as psychology,
    neuroscience, sociology, and physiology, thisdata science project
    takes a different lens. Our primary focus is to delve intothe realm
    of Spotify\'s top-streamed songs, aiming to unveil underlying patterns
    and insights that decode the anatomy of a hit song.</p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">In pursuit of this mission,
    this project will address a series of overarching questions:</p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Musical Characteristics: What key music traits, like tempo,
    dance-friendliness, and energy levels, do the top-streamed songs
    have in common?
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Mode Patterns: Are there discernible trends or patterns in the
    musical modes utilized in top-streamed songs? </p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    - Temporal Popularity: Are there temporal or seasonal trends that
    influence song popularity?</p>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">Through this exploration, we 
    aim to unlock the secrets behind the melodies that captivate global 
    audiences, shedding light on the multifaceted world of music in the 
    digital age. </p>
    """, unsafe_allow_html=True)


with tab2:
    df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

    # Removing specific incorrect data from file 
    target_value = 'BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3'
    df = df[df['streams'] != target_value]
    df['streams'] = df['streams'].astype(int)

    # Creating a new column with standardized values for #streams
    scaler = StandardScaler()
    df['s_streams'] = scaler.fit_transform(df[['streams']])
    
    dataset = st.checkbox('Show summary of songs dataset')
    if dataset == True:
        st.table(df.describe())
        st.markdown("""<p class="font_subtext">Table 1: Spotify\'s 
                    top songs </p>""", unsafe_allow_html=True)

    # Creating a pie chart of keys in the songs
    st.markdown('<p class="font_subheader">Categorical attributes in the dataset</p>', 
                unsafe_allow_html=True)

    key1_name = df['key'].value_counts().index
    key1_value = df['key'].value_counts()

    key2_name = df['mode'].value_counts().index
    key2_value = df['mode'].value_counts()

    fig3, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].pie(key1_value, labels=key1_name, 
                autopct='%1.1f%%', colors= sns.color_palette("Set3", 11))
    axes[0].set_title("Keys in songs")
    axes[1].pie(key2_value, labels=key2_name, 
                autopct='%1.1f%%', colors= sns.color_palette("pastel", 2))
    axes[1].set_title("Modes in songs")
    for ax in axes:
        ax.axis('equal')
    st.pyplot(fig3) 

    # Creating a comparision of beats per minute for different modes
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('bpm:Q', bin=True),
        alt.Y('count():Q', title='Frequency'),
        alt.Color('mode:N')
    ).facet(
        column=alt.Column('mode:O', header=alt.Header(title=None))
    ).configure_axisY(
        grid=True
    )
    st.markdown("""<p class="font_subheader">Frequency plot 
                of beats per minute for modes</p>""", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)


with tab3:
    st.markdown("""Note that the size of the bubbles in
                the following plots correspond to the number of times 
                a particular song was streamed""", unsafe_allow_html=True)
    # Creating a scatterplot of user selected song attribute
    options = st.multiselect('Select 2 song attributes to plot',
                             ['danceability_%', 'valence_%', 'energy_%',
                              'acousticness_%', 'instrumentalness_%', 
                              'liveness_%', 'speechiness_%'], 
                             default=['valence_%', 'energy_%'], 
                             max_selections=2, placeholder="Choose an option", 
                             disabled=False, label_visibility="visible")

    fig1 = px.scatter(df,
        y=options[0],
        x=options[1],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=25,
    )
    st.markdown(f'''<p class="font_subheader">Plot of {options[0]} and 
                {options[1]}</p>''', unsafe_allow_html=True)
    st.plotly_chart(fig1, theme=None, use_container_width=True)

    # Creating a scatterplot of user selected charts
    chart_options = st.multiselect('Select 2 charts to plot', 
                                   ['in_spotify_charts', 'in_apple_charts', 
                                    'in_deezer_charts'], 
                                   default=['in_spotify_charts', 'in_apple_charts'], 
                                   max_selections=2, placeholder="Choose an option", 
                                   disabled=False, label_visibility="visible")

    fig1 = px.scatter(df,
        y=chart_options[0],
        x=chart_options[1],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=25,
    )
    st.markdown(f'''<p class="font_subheader">Plot of {chart_options[0]} and 
                {chart_options[1]}</p>''', unsafe_allow_html=True)
    st.plotly_chart(fig1, theme=None, use_container_width=True)

    # Creating a scatterplot of the month/day variation in streams 
    fig1 = px.scatter(df,
        y=df['released_month'],
        x=df['released_day'],
        size="streams",
        color="mode",
        hover_name="track_name",
        size_max=25,
    )
    st.markdown('''<p class="font_subheader">Plot of the Seasonal variation</p>''', 
                unsafe_allow_html=True)
    st.plotly_chart(fig1, theme=None, use_container_width=True)


with tab4:
    # Creating a regression plot of your selection and streams
    reg_option = st.selectbox('Select the variable for regression plot:', 
                              ['in_spotify_playlists', 'in_spotify_charts', 
                               'in_apple_playlists', 'in_apple_charts', 
                               'in_deezer_charts', 'bpm', 'danceability_%', 
                               'valence_%', 'energy_%', 'acousticness_%', 
                               'instrumentalness_%', 'liveness_%', 'speechiness_%'])

    # Sidebar with slider
    st.sidebar.header("Adjust the selected variable for predicted #streams:")
    new_x = st.sidebar.slider("X Value", min_value=int(df[reg_option].min()), 
                              max_value=int(df[reg_option].max()), 
                              value=int(df[reg_option].mean()))

    X = df[reg_option].values.reshape(-1, 1)
    Y = df['s_streams'].values
    model = LinearRegression()
    model.fit(X, Y)
    predicted_y = model.predict(np.array([[new_x]]))

    y_predicted = np.abs(predicted_y[0]*np.std(df['s_streams'])*np.mean(df['streams']))

    check_mark = st.checkbox('Show predicted number of streams')
    if check_mark==True:
        st.markdown(f'''<p class="font_text">Predicted number of 
                    streams when {reg_option} = {new_x} is: {y_predicted:.2f}</p>''', 
                    unsafe_allow_html=True)
    
    st.markdown(f'''<p class="font_subheader">Regression plot 
                for {reg_option}</p>''', unsafe_allow_html=True)

    # Create a scatter plot with the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(df[reg_option], df['s_streams'], label='Data Points')
    plt.plot(df[reg_option], model.predict(X), color='red', label='Regression Line')
    plt.grid(False)
    plt.xlabel(reg_option)
    plt.ylabel('streams')
    plt.legend()
    st.pyplot(plt)
    st.markdown(f'''<p class="font_text">*Note that the streams values on
                y-axis are standardized values</p>''', unsafe_allow_html=True)
    

with tab5:
    st.markdown('<p class="font_subheader">Musical Characteristics</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""<p class="font_text">
    - It appears there is a positive relationship between the valence and 
    energy of the songs. More songs tend to receive high ratings for these 
    musical attributes.
    """, unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - Songs that achieve higher rankings on Apple charts do not consistently 
    receive higher rankings on Spotify charts. This trend holds true for Apple 
    charts compared to Deezer charts as well. More investigation is required to 
    understand the relationship between Spotify charts and Deezer charts.
    """, unsafe_allow_html=True)

    st.markdown('<p class="font_subheader">Mode Patterns</p>', unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - A clear pattern is coming to light concerning the song modes. In our 
    analysis, we\'ve observed that the beats per minute (BPM) frequency for 
    songs in the Major mode surpasses that of the Minor mode.
    """, unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - From this dataset, it's clear that Major mode songs receive more streams 
    than Minor mode songs. However, it's important to note that this dataset 
    does not have an equal number of songs from each category. Therefore, to 
    make a definitive statement, further investigation with different datasets 
    is necessary.""", unsafe_allow_html=True)

    st.markdown('<p class="font_subheader">Temporal Popularity</p>', 
                unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - It appears that there is no apparent connection between the month or day 
    a song is released and its streaming frequency. This suggests that great 
    music finds its audience year-round, regardless of the release date.
    """, unsafe_allow_html=True)

    st.markdown('<p class="font_subheader">Bottomline</p>', unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    Our analysis revealed several key insights. There appears to be a positive 
    correlation between valence and energy in songs, with many receiving high 
    ratings for these attributes. We also found that high rankings on Apple charts 
    don\'t consistently translate to success on Spotify or Deezer charts, 
    prompting further investigation. Major mode songs tend to have a higher 
    beats per minute (BPM) frequency than Minor mode songs, although data 
    imbalances should be considered. Lastly, the time of a song\'s release 
    doesn\'t significantly affect its streaming frequency, suggesting that 
    music finds an audience year-round.""", unsafe_allow_html=True)
