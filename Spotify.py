import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging

# Configure logging to see more details if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv() # Load your .env file

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI") # Use the corrected one from your .env
SPOTIFY_SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
SPOTIFY_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".spotify_cache")

if __name__ == "__main__":
    print(f"Client ID: {SPOTIPY_CLIENT_ID}")
    print(f"Client Secret: {SPOTIPY_CLIENT_SECRET}")
    print(f"Redirect URI: {SPOTIPY_REDIRECT_URI}")
    print(f"Cache Path: {SPOTIFY_CACHE_PATH}")

    if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET or not SPOTIPY_REDIRECT_URI:
        print("ERROR: Spotify API credentials (SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI) are not fully set in your .env file.")
        exit()

    sp_oauth = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SPOTIFY_SCOPES,
        cache_path=SPOTIFY_CACHE_PATH,
        open_browser=True # IMPORTANT: Let spotipy handle opening the browser
    )

    print("Attempting to get Spotify token...")
    print("A browser window should open for Spotify authentication.")
    print(f"If no browser opens, please manually navigate to: {sp_oauth.get_authorize_url()}")

    try:
        # This call will open the browser, wait for authentication,
        # and automatically handle the redirect to localhost to get the token.
        token_info = sp_oauth.get_access_token(as_dict=True) # Removed check_cache=True as get_access_token is for initial auth

        if token_info:
            print("\nSpotify token obtained and cached successfully!")
            print("You should now find a '.spotify_cache' file in this directory.")
            print(f"Access Token: {'*' * 20}") # Hide sensitive info
            print(f"Refresh Token: {'*' * 20}") # Hide sensitive info
        else:
            print("\nFailed to obtain Spotify token. Please re-check your Spotify Developer Dashboard settings and try again.")
    except Exception as e:
        print(f"\nAn error occurred during Spotify authentication: {e}")
        print("Please ensure:")
        print(f"1. Your Spotify app's Redirect URI is set to: {SPOTIPY_REDIRECT_URI}")
        print("2. You have a stable internet connection.")
        print("3. There are no firewalls or proxies blocking the connection.")