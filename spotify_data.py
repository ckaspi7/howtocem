import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
import os

# Get Spotify credentials from Streamlit secrets
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = st.secrets.get("SPOTIFY_REDIRECT_URI", "https://share.streamlit.io/callback")

# Ensure a writable cache location
CACHE_PATH = "./.spotify_cache"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-top-read user-library-read",
    cache_path=CACHE_PATH
))

def format_top_artists(top_artists):
    """Formats top artists into a structured plain text format."""
    formatted_artists = "\n".join(
        [f"{idx+1}. {artist['name']} - Genres: {', '.join(artist['genres']) if artist['genres'] else 'Unknown'}"
         for idx, artist in enumerate(top_artists)]
    )
    return f"🔹 Top Artists:\n{formatted_artists}"


def format_top_tracks(top_tracks):
    """Formats top tracks into a structured plain text format."""
    formatted_tracks = "\n".join(
        [f"{idx+1}. {track['name']} - {', '.join([artist['name'] for artist in track['artists']])}"
         for idx, track in enumerate(top_tracks)]
    )
    return f"🔹 Top Songs:\n{formatted_tracks}"


def get_top_artists(limit=10, time_range="medium_term"):
    """Fetches the user's top artists and returns a formatted text output."""
    results = sp.current_user_top_artists(limit=limit, time_range=time_range)
    return format_top_artists(results['items'])


def get_top_tracks(limit=10, time_range="medium_term"):
    """Fetches the user's top tracks and returns a formatted text output."""
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    return format_top_tracks(results['items'])

# Only run the example usage if this file is executed directly (not imported)
if __name__ == "__main__":
    formatted_artists = get_top_artists()
    formatted_tracks = get_top_tracks()
    print(formatted_artists)
    print(formatted_tracks)
