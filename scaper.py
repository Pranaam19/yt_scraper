from googleapiclient.discovery import build
import streamlit as st
import pandas as pd
import plotly.express as px
import isodate
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from googleapiclient.errors import HttpError
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()
# YouTube API setup
API_KEY = os.getenv('API_KEY')
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Simple cache implementation
CACHE_FILE = 'youtube_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Function to fetch video details with caching
def get_video_details(video_id):
    cache = load_cache()
    if video_id in cache:
        return cache[video_id]

    request = youtube.videos().list(part='snippet,statistics', id=video_id)
    response = request.execute()
    if response['items']:
        video_info = response['items'][0]
        details = {
            'title': video_info['snippet']['title'],
            'views': int(video_info['statistics'].get('viewCount', 0)),
            'likes': int(video_info['statistics'].get('likeCount', 0)),
            'subscribers': get_channel_subscribers(video_info['snippet']['channelId'])
        }
        cache[video_id] = details
        save_cache(cache)
        return details
    return {}

# Function to fetch channel subscriber count
def get_channel_subscribers(channel_id):
    request = youtube.channels().list(part='statistics', id=channel_id)
    response = request.execute()
    if response['items']:
        return int(response['items'][0]['statistics']['subscriberCount'])
    else:
        return 0

# Function to fetch video IDs from playlist with exponential backoff
def get_video_ids_from_playlist(playlist_id, retries=3):
    video_ids = []
    for attempt in range(retries):
        try:
            request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=50)
            while request:
                response = request.execute()
                for item in response['items']:
                    video_ids.append(item['snippet']['resourceId']['videoId'])
                request = youtube.playlistItems().list_next(request, response)
            return video_ids
        except HttpError as e:
            if e.resp.status == 403 and 'quota' in e.content.decode():
                print("Quota exceeded. Retrying after a delay...")
                time.sleep(60 * (2 ** attempt))  # Exponential backoff
            else:
                print(f"An error occurred: {e}")
                return []
    print(f"Failed to retrieve video IDs from playlist ID: {playlist_id} after {retries} attempts.")
    return []

# Streamlit UI
st.title("YouTube Data Scraper with ML Analysis")
st.write("Enter YouTube video URLs, channel links, or playlist links to get insights and predictions.")

# Input fields for URLs
input_type = st.selectbox("Select Input Type", ["Video URL", "Channel URL", "Playlist URL"])
input_url = st.text_input(f"Enter YouTube {input_type}: ")

if input_url:
    video_data = []

    # Extracting video IDs based on the input type
    if input_type == "Video URL":
        video_ids = [url.strip().split("v=")[-1].split("&")[0] for url in input_url.split(",")]
    elif input_type == "Channel URL":
        if "channel/" in input_url:
            channel_id = input_url.split("channel/")[-1].split("/")[0]
        elif "user/" in input_url:
            username = input_url.split("user/")[-1].split("/")[0]
            try:
                request = youtube.channels().list(part="id", forUsername=username)
                response = request.execute()
                if response['items']:
                    channel_id = response['items'][0]['id']
                else:
                    st.error("Invalid channel username. Please check the URL.")
                    channel_id = None
            except Exception as e:
                st.error(f"Error fetching channel ID: {e}")
                channel_id = None
        else:
            st.error("Invalid Channel URL format. Please use a valid URL format.")
            channel_id = None

        if channel_id:
            try:
                request = youtube.channels().list(part="contentDetails", id=channel_id)
                response = request.execute()
                if response['items']:
                    uploads_playlist = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                    video_ids = get_video_ids_from_playlist(uploads_playlist)
                else:
                    st.error("Unable to find uploads playlist for the channel. It may be private or unavailable.")
                    video_ids = []
            except Exception as e:
                st.error(f"Error fetching uploads playlist: {e}")
                video_ids = []
        else:
            video_ids = []

    elif input_type == "Playlist URL":
        playlist_id = input_url.split("list=")[-1]
        video_ids = get_video_ids_from_playlist(playlist_id)

    for video_id in video_ids:
        details = get_video_details(video_id)
        video_data.append(details)

    df = pd.DataFrame(video_data)

    if not df.empty:
        st.write("### Video Data")
        st.dataframe(df)

        imputer = SimpleImputer(strategy='mean')
        df[['views', 'likes', 'subscribers']] = imputer.fit_transform(df[['views', 'likes', 'subscribers']])

        st.write("### Visualizations")

        fig = px.bar(df, x='title', y=['views', 'likes', 'subscribers'], 
                     title="Comparison of Views, Likes, and Subscribers", 
                     labels={'value': 'Count', 'title': 'Video Title'}, 
                     height=600)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

        # Replace the pie chart with a bar chart for Likes vs Views Distribution
        st.write("### Likes vs Views Distribution")
        fig2 = px.bar(df, x='title', y=['likes', 'views'], 
                      title="Likes and Views Distribution per Video", 
                      labels={'value': 'Count', 'title': 'Video Title'}, 
                      height=400)
        fig2.update_layout(barmode='group')
        st.plotly_chart(fig2)

        fig3 = px.scatter(df, x='views', y='likes', title="Scatter Plot: Views vs Likes", 
                          labels={'views': 'Views', 'likes': 'Likes'})
        st.plotly_chart(fig3)

        st.write("### Linear Regression: Predicting Likes Based on Views")

        X = df[['views']].values
        y = df['likes'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)

        y_pred = linear_reg.predict(X_test)

        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

        fig4 = px.scatter(df, x='views', y='likes', title="Linear Regression: Views vs Likes")
        fig4.add_scatter(x=X_test.flatten(), y=y_pred, mode='lines', name='Regression Line')
        st.plotly_chart(fig4)

        st.write("### Logistic Regression: Classifying High vs Low Engagement Videos")

        threshold = df['likes'].mean()
        df['high_engagement'] = (df['likes'] > threshold).astype(int)

        X = df[['views', 'subscribers']].values
        y = df['high_engagement'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logistic_reg = LogisticRegression()
        logistic_reg.fit(X_train, y_train)

        y_pred = logistic_reg.predict(X_test)

        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        df['predicted_engagement'] = logistic_reg.predict(df[['views', 'subscribers']].values)
        fig5 = px.scatter(df, x='views', y='subscribers', color='predicted_engagement',
                          title="Logistic Regression: Predicted Engagement",
                          labels={'views': 'Views', 'subscribers': 'Subscribers'})
        st.plotly_chart(fig5)

        # Bubble Chart for Video Engagement Metrics
        st.write("### Bubble Chart: Video Engagement Metrics")
        fig_bubble = px.scatter(df, 
                                 x='views', 
                                 y='likes', 
                                 size='subscribers', 
                                 color='title', 
                                 hover_name='title', 
                                 title="Bubble Chart of Video Engagement Metrics",
                                 labels={'views': 'Views', 'likes': 'Likes'},
                                 size_max=60)
        st.plotly_chart(fig_bubble)

        best_video = df.loc[df[['views', 'likes']].sum(axis=1).idxmax()]
        st.write("### Best Video based on Likes and Views")
        st.write(f"Title: {best_video['title']}")
        st.write(f"Views: {best_video['views']}")
        st.write(f"Likes: {best_video['likes']}")
    else:
        st.write("No valid video data found.")