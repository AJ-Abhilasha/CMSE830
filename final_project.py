import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
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

st.markdown(""" <style> .bold_text {
font-size:20px ; font-family: 'sans'; color: black;text-align: left;font-weight: bold;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font_title">The Symphony of Success</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4= st.tabs(["**About this App**", "**Explore the data**", "**Predictions**", "**About me**"])


with tab1:
    st.image("https://i0.wp.com/www.worldtop2.com/wp-content/uploads/2019/11/Top-20-Best-English-Female-pop-Singers-of-2019.jpg?fit=1200%2C675&ssl=1")
    st.markdown("""<p class="font_text">
    Welcome to our music exploration app, blending data science with the magic of music! 
    Explore the secrets behind hit songs, whether you're a music pro seeking insights or 
    a curious enthusiast. Unravel the patterns in beats per minute, danceability, and energy 
    using machine learning to predict a song's 
    hit potential. Our user-friendly interface caters to industry experts and music lovers, 
    revealing the alchemy of musical success. Join us on this musical odyssey where every beat tells a story.
    """, unsafe_allow_html=True)
    

with tab2:
    st.markdown("""<p class="font_text">
    Dive into the intricacies of the dataset, the backbone of our application. 
    Here you can uncover its complexities and gain insights into the wealth of 
    information driving the app's functionalities. Read on for a detailed 
    description of the data.
    """, unsafe_allow_html=True)
    df = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

    # Add a column that is the combination of track name and the artist
    df['track_&_artist'] = df['track_name'] + ' by ' + df['artist(s)_name']

    # Combine day, month, and year columns into a 'date' column
    df['release_date'] = pd.to_datetime(df[['released_day', 'released_month', 'released_year']].rename(columns={'released_day': 'day', 'released_month': 'month', 'released_year': 'year'}))
    df['released_on'] = df['release_date'].dt.date

    # Drop the original day, month, and year columns if needed
    df.drop(columns=['released_year', 'released_month', 'released_day', 'in_spotify_playlists', 'in_spotify_charts', 'in_shazam_charts', 'in_deezer_playlists', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_charts'], inplace=True)

    # Rename the columns
    df = df.rename(columns={'danceability_%': 'danceability', 'valence_%': 'valence', 'energy_%': 'energy', 'acousticness_%': 'acousticness', 'acousticness_%': 'acousticness', 'liveness_%': 'liveness', 'speechiness_%': 'speechiness', 'instrumentalness_%': 'instrumentalness', 'bpm': 'beats_per_minute'})

    # Removing specific incorrect data from file 
    target_value = 'BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3'
    df = df[df['streams'] != target_value]
    df['streams'] = df['streams'].astype(int)

    df['track_name'] = df['track_name'].replace('All Too Well (10 Minute Version) (Taylor\'s Version) (From The Vault)', 'All Too Well')
    df['track_name'] = df['track_name'].replace('Enemy (with JID) - from the series Arcane League of Legends', 'Enemy')

    # Filtering rows that do not contain the "¿" character in any column
    df = df[~df.apply(lambda row: any(row.astype(str).str.contains('¿')), axis=1)]

    # Replace null values in the key column with "other"
    df['key'] = df['key'].fillna('other')

    with st.expander('**Summary of songs dataset**'):
        tab21, tab22 = st.tabs(["Overview of the dataset", "Summary statistics"])
        with tab21:
            st.markdown("""<p class="font_text">This app draws insights from a dataset comprising 847 songs. Along with the name of the artist(s), the release date of the song and 
            number of times the song was streamed, each song has the following attributes:
            """, unsafe_allow_html=True)

            st.markdown("""**🥁 Beats per minute** is the measure of the song's tempo.
            """, unsafe_allow_html=True)
            
            st.markdown("""**💃🏻 Danceability** indicates how suitable the song is for dancing.
            """, unsafe_allow_html=True)
            
            st.markdown("""**😁 Valence** indicates the positivity of the song.
            """, unsafe_allow_html=True)
            
            st.markdown("""**😃 Energy** indicates the perceived energy level of the song.
            """, unsafe_allow_html=True)
            
            st.markdown("""**🎸 Acousticness** indicates the amount of acoustic sound in the song.
            """, unsafe_allow_html=True)
            
            st.markdown("""**🎷 Instrumentalness** indicates the amount of instrumental content in the song.
            """, unsafe_allow_html=True)
            
            st.markdown("""**🎙️ Liveness** indicates the presence of live performance elements.
            """, unsafe_allow_html=True)

            st.markdown("""**📣 Speechiness** indicates the amount of spoken words in the song.
            """, unsafe_allow_html=True)

            st.markdown("""**🎭 Mode** conveys the emotion. Major = happy, minor = sad.
            """, unsafe_allow_html=True) 


        with tab22:
            st.table(df.describe().T)
            st.write("[Click here](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) to view the original data on kaggle.com")

    # Creating a new column with standardized values for #streams
    scaler = StandardScaler()
    df['s_streams'] = scaler.fit_transform(df[['streams']])

    
    with st.expander('**Explore the data**'):

        col1, col2, col3, col4, col5 = st.columns([2, 0.5, 2, 0.5, 2])
        with col1:
            page = st.radio("Filter by:", ["Artist", "Song attributes"])

        with col5:
            number = st.number_input('\# records to show (max 20)', min_value=1, max_value=20, value=5, step=1)
            
        with col3:    
            sort_option = st.selectbox("Sort by", ["--Select--", "Most streamed", "Least streamed", "Latest first", "Oldest first"], index=0)


        if page == "Artist":
            col1, col2, col3 = st.columns([1, 2, 1])
            selected_rows = df
            with col2:
                all_elements = [element for sublist in df['artist(s)_name'].str.split(',') for element in sublist]
                default_value = 'Taylor Swift'
                artist_list = sorted(set(all_elements))
                selected_value = st.selectbox('Select an atrist', artist_list, index=artist_list.index(default_value))

                matching_rows = df[df['artist(s)_name'].str.contains(selected_value, case=False, na=False)]


                if sort_option == "--Select--" or sort_option == "Most streamed":
                    sorted_df = matching_rows.sort_values(by='streams', ascending=False)
                elif sort_option == "Least streamed":
                    sorted_df = matching_rows.sort_values(by='streams', ascending=True)
                elif sort_option ==  "Latest first":
                    sorted_df = matching_rows.sort_values(by='released_on', ascending=False)
                else:
                    sorted_df = matching_rows.sort_values(by='released_on', ascending=True)

            st.write(f'There are {len(sorted_df)} songs by {selected_value} in the dataset.')
            table = sorted_df[["track_name", "released_on", "streams"]].head(number).reset_index(drop=True)
            st.table(table)

        else:
            attributes = ['beats_per_minute', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
            labels = {'beats_per_minute': 'Beats per minute', 'danceability': 'Danceability', 'valence': 'Valence', 'energy': 'Energy', 'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness', 'liveness': 'Liveness', 'speechiness': 'Speechiness'}
            options = st.multiselect('Select the song attributes and adjust their range', attributes, default=['beats_per_minute', 'danceability', 'valence', 'energy'], max_selections=8, placeholder="Choose an option", disabled=False, label_visibility="visible")
            counter = 0
            values = {}
            col1, col2, col3, col4, col5 = st.columns([0.25, 2, 1, 2, 0.25])
            for one_option in options:
                if counter % 2 == 0:
                    with col2:
                        min_val, max_val = st.slider(f"**{labels[one_option]}** range", min(df[one_option]), max(df[one_option]), value = [min(df[one_option]), max(df[one_option])])
                        values[one_option] = [int(min_val), int(max_val)]
                else:
                    with col4:
                        min_val, max_val = st.slider(f"**{labels[one_option]}** range", min(df[one_option]), max(df[one_option]), value = [min(df[one_option]), max(df[one_option])])
                        values[one_option] = [int(min_val), int(max_val)]
                counter = counter + 1

            cols = list(values.keys())
            selected_rows = df
            # Boolean indexing to select rows based on conditions
            for one_col in cols:
                selected_rows = selected_rows[(selected_rows[one_col].between(*values[one_col]))]

                if sort_option == "--Select--" or sort_option == "Most streamed":
                    sorted_df = selected_rows.sort_values(by='streams', ascending=False)
                elif sort_option == "Least streamed":
                    sorted_df = selected_rows.sort_values(by='streams', ascending=True)
                elif sort_option ==  "Latest first":
                    sorted_df = selected_rows.sort_values(by='released_on', ascending=False)
                else:
                    sorted_df = selected_rows.sort_values(by='released_on', ascending=True)

            st.write(f'There are {len(sorted_df)} songs matching the values you selected. Here\'s the top {number} songs.')
            table = sorted_df[["track_name", "artist(s)_name", "released_on", "streams"]].head(number).reset_index(drop=True)
            st.table(table)


        df1 = sorted_df[["track_name", 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'streams', 'track_&_artist', 'mode', 'beats_per_minute']].head(number)
        if not table.empty:   
            button = st.checkbox("Visualize these results")
            if button:
                attributes = ['danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness']

                multi_options = st.multiselect('Select the song attributes', attributes, default=['danceability', 'valence', 'energy', 'liveness'], max_selections=7, placeholder="Choose an option", disabled=False, label_visibility="visible")

                # Create an interactive line plot with hover effects

                fig = px.line(df1, x=df1["track_name"], y=multi_options, labels={'value': 'Percentage'}, line_shape='linear', markers=True)  

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig)

                options = st.multiselect('Select 2 song attributes to plot', ['beats_per_minute', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness'], default=['valence', 'energy'], max_selections=2, placeholder="Choose an option", disabled=False, label_visibility="visible")

                fig1 = px.scatter(df1,
                    y=options[0],
                    x=options[1],
                    size="streams",
                    color="mode",
                    hover_name="track_&_artist",
                    size_max=50,
                )
                st.markdown(f'<p class="font_subheader">Plot of {options[0]} and {options[1]}</p>', unsafe_allow_html=True)
                st.plotly_chart(fig1, theme=None, use_container_width=True)

    
with tab3:
    st.markdown("""<p class="font_text">
    Here, we will predict the number of times your song will be streamed based on various attributes.
    Feel free to experiment with any number of attributes from the list. 
    Tweak each attribute's value using the sliders. Make sure to check the box to uncover the predictions and explore the songs
    that are most similar to yours (at least in terms of qualia). 
    """, unsafe_allow_html=True)
    
    labels = {'beats_per_minute': 'Beats per minute', 'danceability': 'Danceability', 'valence': 'Valence', 'energy': 'Energy', 'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness', 'liveness': 'Liveness', 'speechiness': 'Speechiness'}
            
    options = st.multiselect('Select as many attributes as you wish from the list', ['beats_per_minute', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'speechiness'], default=['beats_per_minute', 'energy', 'valence'], max_selections=7, placeholder="Choose an option", disabled=False, label_visibility="visible")

    counter = 0
    new_x = [0] * len(options)
    for one_option in options:
        new_x[counter] = st.slider(f"Adjust the **{labels[one_option]}**", min_value=int(df[one_option].min()), max_value=int(df[one_option].max()), value=int(df[one_option].mean()))
        counter = counter + 1
    
    X = df[options].values.reshape(-1, len(options))
    Y = df['s_streams'].values
    model = LinearRegression()
    model.fit(X, Y)
    predicted_y = model.predict(np.array([new_x]).reshape(1, -1))

    y_predicted = int(abs(predicted_y[0]*np.std(df['streams'])+np.mean(df['streams'])))

    
    query_df = pd.DataFrame([new_x], columns=options)
    df_selected = df[options]

    # Fit a Nearest Neighbors model
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(df_selected)

    # Find the indices of the 5 closest rows to the query row
    distances, indices = knn.kneighbors(query_df[options])

    # Display the 5 closest rows
    closest_rows = df.iloc[indices[0]]
    table = closest_rows[["track_&_artist", 'released_on', 'streams']].head(number).reset_index(drop=True)
    
    box1 = st.checkbox("Estimate the number of streams")
    if box1:
        st.markdown(f"""<p class="font_text">
        Based on our model, it is predicted that your song will be streamed {y_predicted:,} times!
        """, unsafe_allow_html=True)

        st.markdown(f"""<p class="font_text">
        Here's five records that are most similar to your song.
        """, unsafe_allow_html=True)

        st.table(table)


with tab4:
    
    st.markdown("""<p class="font_text">
    Greetings! I'm Abhilasha, a data science graduate student at Michigan State University. 
    I thrive on uncovering hidden insights within data, particularly in the realms of music, 
    sports, and health. When not immersed in numbers, you'll catch me running or indulging in 
    the artistry of crocheting.
    """, unsafe_allow_html=True)
    st.write("[LinkedIn](https://www.linkedin.com/in/abhilasha-jagtap/)")




