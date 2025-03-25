import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st
import os

# Spotify credentials from Streamlit secrets
# CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
# CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
# REDIRECT_URI = "https://howtocem-test.streamlit.app"

# Use a writable cache location on Streamlit Cloud
# CACHE_PATH = "/tmp/.spotify_cache"

# # Set up Spotify authentication
# sp_oauth = SpotifyOAuth(
#     client_id=CLIENT_ID,
#     client_secret=CLIENT_SECRET,
#     redirect_uri=REDIRECT_URI,
#     scope="user-top-read user-library-read"
# )

# # Check if the user is authenticated
# token_info = sp_oauth.get_cached_token()

# if not token_info:
#     auth_url = sp_oauth.get_authorize_url()
#     st.markdown(f"[Click here to authorize with Spotify]({auth_url})")
# else:
#     access_token = token_info['access_token']
#     sp = spotipy.Spotify(auth=access_token)
#     st.success("Successfully authenticated with Spotify!")

def get_top_artists(limit=10, time_range="medium_term"):
    """Fetches user's top artists."""
    return [
        {"rank": 1, "name": "The Weeknd", "genres": "Unknown"},
        {"rank": 2, "name": "Kendrick Lamar", "genres": "hip hop, west coast hip hop"},
        {"rank": 3, "name": "Future", "genres": "rap"},
        {"rank": 4, "name": "Drake", "genres": "rap, hip hop"},
        {"rank": 5, "name": "Bad Bunny", "genres": "reggaeton, trap latino, latin, urbano latino"},
        {"rank": 6, "name": "Aaryan Shah", "genres": "Unknown"},
        {"rank": 7, "name": "Eminem", "genres": "rap, hip hop"},
        {"rank": 8, "name": "GIMS", "genres": "french pop, pop urbaine"},
        {"rank": 9, "name": "2Pac", "genres": "gangster rap, west coast hip hop, g-funk, hip hop"},
        {"rank": 10, "name": "deadmau5", "genres": "edm, progressive house, dubstep"}
    ]
    # if not sp:
    #     return "Spotify connection not available."
    # try:
    #     results = sp.current_user_top_artists(limit=limit, time_range=time_range)
    #     return "\n".join([f"{i+1}. {artist['name']}" for i, artist in enumerate(results['items'])])
    # except Exception as e:
    #     return f"Error fetching top artists: {e}"

def get_top_tracks(limit=10, time_range="medium_term"):
    """Fetches user's top tracks."""
    return [
        {"rank": 1, "title": "Digital Dash", "artists": "Drake, Future"},
        {"rank": 2, "title": "Wavy", "artists": "Karan Aujla"},
        {"rank": 3, "title": "VOY A LLeVARTE PA PR", "artists": "Bad Bunny"},
        {"rank": 4, "title": "Given Up On Me", "artists": "The Weeknd"},
        {"rank": 5, "title": "SEQUÊNCIA MALÉFICA 1.0", "artists": "RXPOSO99, Mc Delux"},
        {"rank": 6, "title": "Sweat", "artists": "DJSM"},
        {"rank": 7, "title": "Cry For Me", "artists": "The Weeknd"},
        {"rank": 8, "title": "A Tale Of 2 Citiez", "artists": "J. Cole"},
        {"rank": 9, "title": "EEYUH! x Fluxxwave - Slowed + Reverb", "artists": "Clovis Reyes, HR, Irokz"},
        {"rank": 10, "title": "Tak Tak Funk - Slowed", "artists": "chipbagov, SCARIONIX, IMMORTAL PLAYA"}
    ]
    # if not sp:
    #     return "Spotify connection not available."
    # try:
    #     results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    #     return "\n".join([f"{i+1}. {track['name']} - {', '.join(a['name'] for a in track['artists'])}" for i, track in enumerate(results['items'])])
    # except Exception as e:
    #     return f"Error fetching top tracks: {e}"




# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
# import streamlit as st
# import os

# # Get Spotify credentials from Streamlit secrets
# CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
# CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
# REDIRECT_URI = "https://howtocem-test.streamlit.app/callback"

# # Ensure a writable cache location
# CACHE_PATH = "/tmp/.spotify_cache"

# # Print diagnostics
# print(f"Spotify cache path: {os.path.abspath(CACHE_PATH)}")
# print(f"Spotify cache exists: {os.path.exists(CACHE_PATH)}")
# print(f"Spotify redirect URI: {REDIRECT_URI}")

# # Authenticate with Spotify
# try:
#     sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#         client_id=CLIENT_ID,
#         client_secret=CLIENT_SECRET,
#         redirect_uri=REDIRECT_URI,
#         scope="user-top-read user-library-read",
#         cache_path=CACHE_PATH,
#         open_browser=False
#     ))
#     # Test connection
#     sp.current_user()
#     print("Spotify authentication successful")
# except Exception as e:
#     print(f"Spotify authentication error: {str(e)}")
#     sp = None

# def format_top_artists(top_artists):
#     """Formats top artists into a structured plain text format."""
#     formatted_artists = "\n".join(
#         [f"{idx+1}. {artist['name']} - Genres: {', '.join(artist['genres']) if artist['genres'] else 'Unknown'}"
#          for idx, artist in enumerate(top_artists)]
#     )
#     return f"🔹 Top Artists:\n{formatted_artists}"


# def format_top_tracks(top_tracks):
#     """Formats top tracks into a structured plain text format."""
#     formatted_tracks = "\n".join(
#         [f"{idx+1}. {track['name']} - {', '.join([artist['name'] for artist in track['artists']])}"
#          for idx, track in enumerate(top_tracks)]
#     )
#     return f"🔹 Top Songs:\n{formatted_tracks}"


# def get_top_artists(limit=10, time_range="medium_term"):
#     """Fetches the user's top artists and returns a formatted text output."""
#     if sp is None:
#         return "Spotify connection not available. Please check authentication."
        
#     try:
#         results = sp.current_user_top_artists(limit=limit, time_range=time_range)
#         return format_top_artists(results['items'])
#     except Exception as e:
#         return f"Error fetching top artists: {str(e)}"

# def get_top_tracks(limit=10, time_range="medium_term"):
#     """Fetches the user's top tracks and returns a formatted text output."""
#     if sp is None:
#         return "Spotify connection not available. Please check authentication."
#     try:
#         results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
#         return format_top_tracks(results['items'])
#     except Exception as e:
#         return f"Error fetching top tracks: {str(e)}"

# Only run the example usage if this file is executed directly (not imported)
if __name__ == "__main__":
    # formatted_artists = get_top_artists()
    # formatted_tracks = get_top_tracks()
    # print(formatted_artists)
    # print(formatted_tracks)
    print(get_top_artists())
    print(get_top_tracks())
