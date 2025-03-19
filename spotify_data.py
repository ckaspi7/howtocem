import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from a .env file

# Get your Spotify credentials from the .env file
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Ensure a writable cache location
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".spotify_cache")

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri="http://127.0.0.1:5000/callback",
    scope="user-top-read user-library-read"
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


# Example Usage
formatted_artists = get_top_artists()
formatted_tracks = get_top_tracks()

print(formatted_artists)
print(formatted_tracks)