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
font-size:40px; font-family:'courier'; color:black; text-align:center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:30px ; font-family: 'times'; color: black;text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:20px ; font-family: 'sans'; color: black;text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font_title">The Symphony of Success</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**Introduction**", "**Data and variables**", "**Plots**", "**Predictions**", "**Remarks**", "**Bio**"])


with tab1:
    st.markdown("""<p class="font_text">
    Welcome to our innovative music exploration app, where the realms of data science meet the enchanting world of music! Uncover the magic behind hit songs as we delve into key musical attributes that shape chart-topping tunes. Whether you're a seasoned music industry professional seeking data-driven insights or a curious enthusiast eager to understand the secrets of a hit, this app is your gateway to a melodic journey.
    """, unsafe_allow_html=True)
    
    st.markdown("""<p class="font_text">
    By harnessing the power of machine learning, we analyze song attributes such as beats per minute, danceability, and energy, unraveling the intricate patterns that contribute to a song's success. Walk in the footsteps of music professionals, exploring the art and science behind what makes a song resonate with audiences.
    """, unsafe_allow_html=True)
    
    st.markdown("""<p class="font_text">
    From the heartbeat of BPM to the rhythm of danceability, embark on a predictive adventure to discern whether a song possesses the magic to become a hit. Our user-friendly interface caters to both industry experts and music lovers alike, providing valuable insights into the alchemy of musical success.
    """, unsafe_allow_html=True)
    
    st.markdown("""<p class="font_text">
    Whether you're crafting the next chart-topper or simply savoring the essence of a potential hit, join us on this musical odyssey where data meets melody, and every beat has a story to tell.
    """, unsafe_allow_html=True)


with tab2:
    st.markdown("""<p class="font_text">
    Explore the intricacies of the dataset that forms the backbone of this application. Delve into a  
    description of the underlying data, unraveling its complexities and gaining insights into the wealth 
    of information that fuels the functionalities of this app.
    """, unsafe_allow_html=True)
    df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')
    df.drop(columns=['released_year', 'released_month', 'released_day'], inplace=True)

    # Removing specific incorrect data from file 
    target_value = 'BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3'
    df = df[df['streams'] != target_value]
    df['streams'] = df['streams'].astype(int)

    # Creating a new column with standardized values for #streams
    scaler = StandardScaler()
    df['s_streams'] = scaler.fit_transform(df[['streams']])

    with st.expander('**Summary of songs dataset**'):
        tab21, tab22 = st.tabs(["Overview of the dataset", "Summary statistics"])
        with tab21:
            st.markdown("""<p class="font_text">
            This app draws insights from a robust dataset available on Kaggle.com comprising 952 songs available on Spotify. 
            Within the dataset, you'll find 14 informative columns, including key metrics like the number of song streams, 
            artists, appearances on playlists across platforms, chart rankings, beats per minute, and various song 
            attributes such as danceability, valence, and energy. Together, these facets provide a comprehensive 
            view, allowing for an in-depth exploration of the musical landscape hosted on Spotify.
            """, unsafe_allow_html=True)

        with tab22:
            st.table(df.describe().T)
            st.markdown('<p class="font_subtext">Table 1: Spotify\'s top songs </p>', unsafe_allow_html=True)


    with st.expander('**Key composition overview**'):
        # Creating a pie chart for 'key' attribute
        st.markdown("""<p class="font_text">
            In music, a "key" refers to the tonal center or the central note around which a musical piece revolves. 
            The key provides a sense of tonal stability and determines the relationships between the different 
            pitches in a piece of music. In the dataset used for building this app, 14% of the songs were written in C#.
            """, unsafe_allow_html=True)

        key_name = df['key'].value_counts().index
        key_value = df['key'].value_counts()

        fig_key, ax_key = plt.subplots(figsize=(5, 5))
        ax_key.pie(key_value, labels=key_name, autopct='%1.1f%%', colors=sns.color_palette("Set3", 11))
        #ax_key.set_title("Keys in Songs")
        ax_key.axis('equal')
        st.pyplot(fig_key)

    with st.expander('**Mode composition overview**'):
        st.markdown("""<p class="font_text">
        Modes are essential for creating varied musical compositions, offering different tonalities and emotions. They are used in a wide range of genres, providing musicians with a diverse palette for artistic expression. Major mode is often used to convey feelings of joy, triumph, or celebration. Minor mode is commonly used to express emotions such as sadness, contemplation, or tension.
        """, unsafe_allow_html=True)

        mode_name = df['mode'].value_counts().index
        mode_value = df['mode'].value_counts()

        fig_mode, ax_mode = plt.subplots(figsize=(5, 5))
        ax_mode.pie(mode_value, labels=mode_name, autopct='%1.1f%%', colors=sns.color_palette("pastel", 2))
        #ax_mode.set_title("Modes in Songs")
        ax_mode.axis('equal')
        st.pyplot(fig_mode)

    with st.expander('**Distribution of beats per minute**'):
        st.markdown("""<p class="font_text">
        Beats per minute (BPM) is a measure of the tempo or speed of a piece of music. It represents the number of beats (pulses or taps) in one minute. BPM is a crucial element in music, influencing the overall feel, energy, and pacing of a composition.
        A higher BPM generally corresponds to a faster tempo, creating a sense of urgency or excitement. Conversely, a lower BPM indicates a slower tempo, often associated with more relaxed or contemplative moods.
        """, unsafe_allow_html=True)
        
        # bpm chart
        chart = alt.Chart(df).mark_bar().encode(
            alt.X('bpm:Q', bin=True),
            alt.Y('count():Q', title='Frequency')
        )
        st.altair_chart(chart, use_container_width=False)


with tab3:
    st.markdown("""<p class="font_text">
    Ever wonder about the subtle influence of major or minor modes in the world of chart-topping tunes? 
    Now, you can visualize it. Select two song attributes or explore the song charts where these melodies 
    make their mark. The dropdown menu within each section below offers an array of options for you 
    to delve into and fascinating interplay between mode types and streaming frequencies. Each bubble in 
    the plot mirrors a song's streaming frequency – the bigger the bubble, the more hearts and ears it has embraced.  
    """, unsafe_allow_html=True)

    # Creating the expander for song attributes
    with st.expander("**Explore song attributes**"):
        options = st.multiselect('Select 2 song attributes to plot', ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'], default=['valence_%', 'energy_%'], max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

        fig1 = px.scatter(df,
            y=options[0],
            x=options[1],
            size="streams",
            color="mode",
            hover_name="track_name",
            size_max=25,
        )
        st.markdown(f'<p class="font_subheader">Plot of {options[0]} and {options[1]}</p>', unsafe_allow_html=True)
        st.plotly_chart(fig1, theme=None, use_container_width=True)

    #st.markdown("Note that the size of the bubbles in the following plots correspond to the number of times a particular song was streamed")
    
    # Creating a scatterplot of user selected charts within the expander for streaming platforms
    with st.expander("**Explore songs on different charts**"):
        chart_options = st.multiselect('Select 2 charts to plot', ['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts'], default=['in_spotify_charts', 'in_apple_charts'], max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

        fig1 = px.scatter(df,
            y=chart_options[0],
            x=chart_options[1],
            size="streams",
            color="mode",
            hover_name="track_name",
            size_max=25,
        )
        st.markdown(f'<p class="font_subheader">Plot of {chart_options[0]} and {chart_options[1]}</p>', unsafe_allow_html=True)
        st.plotly_chart(fig1, theme=None, use_container_width=True)

    

with tab4:
    st.markdown("""<p class="font_text">
    Here, we embark on predicting how many times your song will be streamed, driven by your chosen song attribute(s) 
    and their respective values. Feel free to experiment with any number of attributes from the list. 
    Tweak each attribute's value using the sliders to uncover the diverse predictions and explore the dynamic 
    impact of these features on your song's streaming potential.
    """, unsafe_allow_html=True)
    
    #st.markdown(f'<p class="font_subheader">Alright! Select as many attributes as you wish from the list.</p>', unsafe_allow_html=True)
    options = st.multiselect('Select as many attributes as you wish from the list', ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'speechiness_%'], default=['bpm'], max_selections=7, placeholder="Choose an option", disabled=False, label_visibility="visible")

    counter = 0
    new_x = [0] * len(options)
    for one_option in options:
        new_x[counter] = st.slider(f"Adjust the {one_option}", min_value=int(df[one_option].min()), max_value=int(df[one_option].max()))
        counter = counter + 1

    X = df[options].values.reshape(-1, len(options))
    Y = df['s_streams'].values
    model = LinearRegression()
    model.fit(X, Y)
    predicted_y = model.predict(np.array([new_x]).reshape(1, -1))

    y_predicted = int(abs(predicted_y[0]*np.std(df['streams'])+np.mean(df['streams'])))

    done_button = st.button("I'm Done", help="Click to show results")
    if done_button:
        st.markdown(f"""<p class="font_text">
        Predicted number of streams with the given values is: {y_predicted:,}
        """, unsafe_allow_html=True)


with tab5:
    st.markdown('<p class="font_subheader">Musical Characteristics</p>', unsafe_allow_html=True)
    
    st.markdown("""<p class="font_text">
    - It appears there is a positive relationship between the valence and energy of the songs. 
    More songs tend to receive high ratings for these musical attributes.
    """, unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - Songs that achieve higher rankings on Apple charts do not consistently receive higher rankings on Spotify charts. 
    This trend holds true for Apple charts compared to Deezer charts as well.
    """, unsafe_allow_html=True)

    st.markdown('<p class="font_subheader">Mode Patterns</p>', unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - The beats per minute (BPM) for a song in the Major mode surpasses that of the Minor mode.
    """, unsafe_allow_html=True)
    st.markdown("""<p class="font_text">
    - Major mode songs receive more streams than Minor mode songs. 
    However, it's important to note that this dataset does not have an equal number of songs from each category. 
    Therefore, to make a definitive statement, further investigation with different datasets is necessary.
    """, unsafe_allow_html=True)

with tab6:
    st.markdown("""<p class="font_text">
    Hey there! I'm Abhilasha, a graduate student immersed in the fascinating world of data science at Michigan State
    University. I'm passionate about unraveling insights hidden within data, especially in the realms of music, 
    sports, and health. When I'm not crunching numbers, I find solace in the rhythm of my runs and the artistry of crocheting.
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    I have created this Streamlit app with the goal to furnish music industry professionals with the transformative 
    power of data-driven insights. Picture it as a symphony of success, where the precision of data harmonizes 
    seamlessly with the boundless creativity of the industry.
    """, unsafe_allow_html=True)

    st.markdown("""<p class="font_text">
    Should you have any queries or wish to share your thoughts on this musical endeavor, don't hesitate to reach out 
    at abhilashajagtap2@gmail.com. Let's orchestrate success together. Cheers!
    """, unsafe_allow_html=True)



