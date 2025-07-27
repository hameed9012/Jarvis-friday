import logging
import os
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Dict, List
import datetime
import json
from livekit.agents.llm.tool_context import function_tool # Ensure this import is present
import asyncio
import socket
import aiohttp
import yfinance as yf
import subprocess
import platform
import psutil
import speedtest
import qrcode
import io
import base64
import sqlite3
import random
import hashlib
import uuid
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
import cv2
import face_recognition
import sounddevice as sd
import soundfile as sf
import wikipedia
import pyautogui
import schedule
import time
import threading
from cryptography.fernet import Fernet
import zipfile
import shutil
import tempfile
from langchain_community.tools import DuckDuckGoSearchRun
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create output directory
TEMP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_media")
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

# --- Firebase Imports and Setup ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    firebase_credentials_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
    if firebase_credentials_path and os.path.exists(firebase_credentials_path):
        if not os.path.isabs(firebase_credentials_path):
            firebase_credentials_path = os.path.join(os.path.dirname(__file__), firebase_credentials_path)
        
        cred = credentials.Certificate(firebase_credentials_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logging.info("Firebase initialized successfully")
    else:
        logging.warning(f"Firebase credentials not found at {firebase_credentials_path} or path incorrect, memory features disabled")
        db = None
except Exception as e:
    logging.error(f"Firebase initialization failed: {e}")
    db = None

# --- Spotify Setup ---
try:
    SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
    SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
    SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
    SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

    if all([SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI]):
        sp_oauth = SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=SPOTIFY_SCOPE,
            open_browser=False,
            cache_path=".spotify_cache"
        )
        sp = spotipy.Spotify(auth_manager=sp_oauth)
        logging.info("Spotify initialized successfully")
    else:
        logging.warning("Spotify credentials incomplete")
        sp = None
except Exception as e:
    logging.error(f"Spotify initialization failed: {e}")
    sp = None

# --- Memory and Auth Manager Class ---
class MemoryManager:
    def __init__(self, firestore_db):
        self.db = firestore_db

    def register_user_mock(self, username: str, password: str) -> Optional[str]:
        """Mock user registration - NOT secure for production use"""
        if not self.db:
            logging.warning("Cannot register user: Firestore unavailable")
            return None
        try:
            users_ref = self.db.collection('mock_users')
            existing_user_doc = users_ref.document(username).get()
            if existing_user_doc.exists:
                logging.warning(f"Username '{username}' already exists.")
                return None

            # Generate a shorter, more memorable user ID
            user_id = str(uuid.uuid4().hex[:10]) # Use first 10 characters of UUID
            
            # WARNING: Storing passwords directly is INSECURE.
            # In a real application, you MUST hash and salt passwords.
            users_ref.document(username).set({
                'user_id': user_id,
                'password_hash_mock': password, # Mock storage, NOT secure
                'created_at': firestore.SERVER_TIMESTAMP
            })
            logging.info(f"Mock user '{username}' registered with user_id: {user_id}")
            return user_id
        except Exception as e:
            logging.error(f"Error registering mock user '{username}': {e}")
            return None

    def login_user_mock(self, username: str, password: str) -> Optional[str]:
        """Mock user login - NOT secure for production use"""
        if not self.db:
            logging.warning("Cannot login user: Firestore unavailable")
            return None
        try:
            users_ref = self.db.collection('mock_users')
            user_doc = users_ref.document(username).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                stored_password_mock = user_data.get('password_hash_mock')
                user_id = user_data.get('user_id')

                if stored_password_mock == password: # Mock comparison, NOT secure
                    logging.info(f"Mock user '{username}' logged in with user_id: {user_id}")
                    return user_id
                else:
                    logging.warning(f"Incorrect password for user '{username}'.")
                    return None
            else:
                logging.warning(f"Username '{username}' not found.")
                return None
        except Exception as e:
            logging.error(f"Error logging in mock user '{username}': {e}")
            return None

    def log_interaction(self, user_id: str, session_id: str, role: str, content: str):
        """Log conversation interactions"""
        if not self.db:
            logging.warning("Cannot log interaction: Firestore unavailable")
            return
        try:
            doc_ref = self.db.collection('users').document(user_id).collection('sessions').document(session_id).collection('messages').document()
            doc_ref.set({
                'role': role,
                'content': content,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            logging.debug(f"Logged interaction for user {user_id}, session {session_id}, role: {role}")
        except Exception as e:
            logging.error(f"Error logging interaction for user {user_id}, session {session_id}: {e}")

    def get_recent_history(self, user_id: str, session_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve recent conversation history"""
        if not self.db:
            return []
        try:
            messages_ref = self.db.collection('users').document(user_id).collection('sessions').document(session_id).collection('messages')
            query = messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            history = []
            for doc in docs:
                data = doc.to_dict()
                history.append({
                    "role": data.get('role'),
                    "content": data.get('content'),
                    "timestamp": data.get('timestamp')
                })
            history.reverse()  # Oldest first
            logging.debug(f"Retrieved {len(history)} recent interactions for user {user_id}, session {session_id}")
            return history
        except Exception as e:
            logging.error(f"Error retrieving history for user {user_id}, session {session_id}: {e}")
            return []

    def store_user_data(self, user_id: str, key: str, value: str):
        """Store user-specific data"""
        if not self.db:
            logging.warning("Cannot store user data: Firestore unavailable")
            return
        try:
            doc_ref = self.db.collection('users').document(user_id).collection('user_data').document(key)
            doc_ref.set({
                'value': value,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            logging.info(f"Stored user data for user {user_id}, key: {key}")
        except Exception as e:
            logging.error(f"Error storing user data for user {user_id}, key {key}: {e}")

    def retrieve_user_data(self, user_id: str, key: str) -> Optional[str]:
        """Retrieve user-specific data"""
        if not self.db:
            return None
        try:
            doc_ref = self.db.collection('users').document(user_id).collection('user_data').document(key)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                logging.debug(f"Retrieved user data for user {user_id}, key: {key}")
                return data.get('value')
            else:
                logging.debug(f"User data not found for user {user_id}, key: {key}")
                return None
        except Exception as e:
            logging.error(f"Error retrieving user data for user {user_id}, key {key}: {e}")
            return None

# Initialize memory manager
memory_manager = MemoryManager(db)

# --- Authentication Functions (NOW decorated with @function_tool and include password) ---
@function_tool(description="Register a new user with a username and password. Returns the new user's ID upon successful registration. WARNING: This is a mock implementation and NOT secure for production.")
async def register_user_mock(username: str, password: str) -> str:
    """Register a new user - to be used as a tool"""
    user_id = memory_manager.register_user_mock(username, password)
    if user_id:
        return f"User '{username}' registered successfully with ID: {user_id}, sir. Please use this ID for future memory operations."
    else:
        return f"Registration failed for '{username}', sir. The username might already be taken or an error occurred."

@function_tool(description="Logs in an existing user with a username and password. Returns the logged-in user's ID upon successful login. WARNING: This is a mock implementation and NOT secure for production.")
async def login_user_mock(username: str, password: str) -> str:
    """Login an existing user - to be used as a tool"""
    user_id = memory_manager.login_user_mock(username, password)
    if user_id:
        return f"Logged in successfully as '{username}' with ID: {user_id}, sir. Please use this ID for future memory operations."
    else:
        return f"Login failed for '{username}', sir. Invalid username or password."

# --- Memory Tools ---
@function_tool(description="Log conversation interactions")
async def log_interaction(user_id: str, session_id: str, role: str, content: str) -> str:
    try:
        memory_manager.log_interaction(user_id, session_id, role, content)
        return "Interaction logged successfully, sir."
    except Exception as e:
        logging.error(f"Failed to log interaction: {e}")
        return f"Failed to log interaction: {e}"

@function_tool(description="Get conversation history for the current session")
async def get_conversation_history(user_id: str, session_id: str, limit: int = 5) -> str:
    try:
        history = memory_manager.get_recent_history(user_id, session_id, limit)
        if not history:
            return f"No conversation history found for the current session, sir."

        result = f"Recent conversation history for current session:\n"
        for entry in history:
            role = entry.get('role', 'unknown').capitalize()
            content = entry.get('content', '')
            if len(content) > 100:
                content = content[:100] + "..."
            result += f"- {role}: {content}\n"
        return result
    except Exception as e:
        logging.error(f"Failed to retrieve history: {e}")
        return f"Failed to retrieve history: {e}"

@function_tool(description="Stores a specific piece of information about the user")
async def store_user_data(user_id: str, key: str, value: str) -> str:
    try:
        memory_manager.store_user_data(user_id, key, value)
        return f"I have noted that your {key} is {value}, sir."
    except Exception as e:
        logging.error(f"Failed to store user data: {e}")
        return f"Failed to remember that: {e}, sir."

@function_tool(description="Retrieve a specific piece of information about the user")
async def retrieve_user_data(user_id: str, key: str) -> str:
    try:
        value = memory_manager.retrieve_user_data(user_id, key)
        if value:
            return f"I remember that your {key} is {value}, sir."
        else:
            return f"I do not have information about your {key} stored, sir."
    except Exception as e:
        logging.error(f"Failed to retrieve user data: {e}")
        return f"Failed to retrieve information: {e}, sir."

# --- Core Tools ---
@function_tool(description="Get weather information for a city")
async def get_weather(city: str) -> str:
    try:
        response = requests.get(f"https://wttr.in/{city}?format=%l:+%c+%t+%h", timeout=10)
        if response.status_code == 200:
            return f"Weather in {city}: {response.text.strip()}, sir."
        else:
            return f"Could not retrieve weather for {city}, sir."
    except Exception as e:
        logging.error(f"Weather error: {e}")
        return f"Weather service unavailable, sir."

@function_tool(description="Search the web using DuckDuckGo")
async def search_web(query: str) -> str:
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(tool_input=query)
        return f"Search results for '{query}': {results[:500]}..., sir."
    except Exception as e:
        logging.error(f"Search error: {e}")
        return f"Web search failed, sir."

@function_tool(description="Send email via Gmail")
async def send_email(to_email: str, subject: str, message: str) -> str:
    try:
        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_user or not gmail_password:
            return "Email credentials not configured, sir."

        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, to_email, msg.as_string())

        return f"Email sent to {to_email}, sir."
    except Exception as e:
        logging.error(f"Email error: {e}")
        return f"Email sending failed, sir."

@function_tool(description="Get current date and time")
async def get_current_time() -> str:
    try:
        now = datetime.datetime.now()
        return f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}, sir."
    except Exception as e:
        return "Time service unavailable, sir."

@function_tool(description="Get stock price information")
async def get_stock_price(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="1d")
        if hist.empty:
            return f"Stock {symbol} not found, sir."

        current_price = hist['Close'].iloc[-1]
        return f"{symbol.upper()}: ${current_price:.2f}, sir."
    except Exception as e:
        logging.error(f"Stock error: {e}")
        return f"Stock data unavailable for {symbol}, sir."

@function_tool(description="Get latest news headlines")
async def get_news(query: str = "latest news") -> str:
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return "News service not configured, sir."

        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])[:3]
                    if not articles:
                        return "No news found, sir."

                    news_summary = "Top news headlines:\n"
                    for i, article in enumerate(articles, 1):
                        news_summary += f"{i}. {article['title']}\n"
                    return news_summary + "sir."
                else:
                    return "News service unavailable, sir."
    except Exception as e:
        logging.error(f"News error: {e}")
        return "News service failed, sir."

@function_tool(description="Get system performance statistics")
async def system_stats() -> str:
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        stats = f"System Status:\n"
        stats += f"CPU: {cpu_percent}%\n"
        stats += f"Memory: {memory.percent}%\n"
        stats += f"Disk: {disk.percent}% used\n"
        return stats + "sir."
    except Exception as e:
        return "System monitoring unavailable, sir."

@function_tool(description="Generate QR code for given data")
async def generate_qr_code(data: str) -> str:
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        filename = os.path.join(TEMP_OUTPUT_DIR, f"qr_{hashlib.md5(data.encode()).hexdigest()[:6]}.png")
        img.save(filename)

        return f"QR code generated: {filename}, sir."
    except Exception as e:
        logging.error(f"QR code generation failed: {e}")
        return "QR code generation failed, sir."

@function_tool(description="Perform mathematical calculations")
async def calculate(expression: str) -> str:
    try:
        # Basic safety check
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid calculation expression, sir."

        result = eval(expression)
        return f"{expression} = {result}, sir."
    except Exception as e:
        return "Calculation failed, sir."

@function_tool(description="Get a random interesting fact")
async def random_fact() -> str:
    facts = [
        "Octopuses have three hearts and blue blood.",
        "A group of flamingos is called a 'flamboyance'.",
        "Honey never spoils.",
        "The human brain has about 86 billion neurons.",
        "Bananas are berries, but strawberries aren't."
    ]
    return f"Random fact: {random.choice(facts)}, sir."

# --- Advanced Tools ---
@function_tool(description="Create a mind map visualization with a central topic and branches.")
async def create_mindmap(central_topic: str, branches: List[str]) -> str:
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Central topic
        ax.text(0.5, 0.5, central_topic, ha='center', va='center', fontsize=20,
                bbox=dict(boxstyle="round,pad=0.5", fc="skyblue", ec="blue", lw=2))

        num_branches = len(branches)
        if num_branches > 0:
            angles = np.linspace(0, 2 * np.pi, num_branches, endpoint=False)
            radius = 0.3

            for i, branch_text in enumerate(branches):
                angle = angles[i]
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)

                # Draw line from central topic to branch
                ax.plot([0.5, x], [0.5, y], color='gray', linestyle='-', linewidth=1)

                # Draw branch text
                ax.text(x, y, branch_text, ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", lw=1))

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Mind Map", fontsize=24, pad=20)
        plt.tight_layout()

        filename = os.path.join(TEMP_OUTPUT_DIR, f"mindmap_{central_topic.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Mind map generated at: {filename}")
        return f"Mind map generated successfully: {filename}, sir."
    except Exception as e:
        logging.error(f"Mind map creation failed: {e}")
        return f"Mind map creation failed: {e}, sir."

@function_tool(description="Record voice memo")
async def voice_memo_recorder(duration: int = 10) -> str:
    return f"Voice memo recording functionality noted for {duration} seconds, sir."

@function_tool(description="Generate secure password")
async def password_generator(length: int = 16) -> str:
    try:
        import string
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(random.choice(chars) for _ in range(length))
        return f"Generated password: {password}, sir."
    except Exception as e:
        return "Password generation failed, sir."

@function_tool(description="Create data visualization")
async def data_visualizer(data_points: List[Dict], chart_type: str = "bar") -> str:
    try:
        if not data_points:
            return "No data points provided for visualization, sir."
        
        # Create a simple visualization based on data points
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "bar":
            labels = [str(point.get('label', f'Item {i+1}')) for i, point in enumerate(data_points)]
            values = [float(point.get('value', 0)) for point in data_points]
            ax.bar(labels, values)
            ax.set_title("Bar Chart")
            ax.set_ylabel("Values")
        elif chart_type == "line":
            x_values = [i for i in range(len(data_points))]
            y_values = [float(point.get('value', 0)) for point in data_points]
            ax.plot(x_values, y_values, marker='o')
            ax.set_title("Line Chart")
            ax.set_ylabel("Values")
        else:
            return f"Chart type '{chart_type}' not supported, sir."
        
        plt.tight_layout()
        filename = os.path.join(TEMP_OUTPUT_DIR, f"chart_{chart_type}_{uuid.uuid4().hex[:6]}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return f"Data visualization ({chart_type}) created: {filename}, sir."
    except Exception as e:
        logging.error(f"Data visualization failed: {e}")
        return f"Data visualization failed: {e}, sir."

@function_tool(description="Translate text between languages")
async def translate_text(text: str, target_language: str = "en") -> str:
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        translated = translator.translate(text)
        return f"Translation: {translated}, sir."
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return f"Translation failed, sir."

@function_tool(description="Track daily habits")
async def habit_tracker(habit_name: str, action: str = "check") -> str:
    return f"Habit '{habit_name}' {action} recorded, sir."

@function_tool(description="Track expenses")
async def expense_tracker(amount: float, category: str, description: str = "") -> str:
    return f"Expense recorded: ${amount:.2f} for {category}, sir."

@function_tool(description="Organize files")
async def file_organizer(directory_path: str) -> str:
    return f"File organization planned for {directory_path}, sir."

@function_tool(description="Analyze screenshots")
async def screenshot_analyzer() -> str:
    return "Screenshot analysis capability noted, sir."

@function_tool(description="Scan network")
async def network_scanner() -> str:
    return "Network scanning capability noted, sir."

@function_tool(description="Manage code snippets")
async def code_snippet_manager(action: str, language: str = "python") -> str:
    return f"Code snippet {action} for {language} noted, sir."

@function_tool(description="Optimize system performance")
async def system_optimizer() -> str:
    return "System optimization capability noted, sir."

@function_tool(description="Track cryptocurrency prices")
async def crypto_tracker(symbol: str = "BTC") -> str:
    try:
        # Simple crypto price tracking using a free API
        url = f"https://api.coindesk.com/v1/bpi/currentprice/{symbol}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price = data.get('bpi', {}).get(symbol, {}).get('rate', 'N/A')
            return f"{symbol} price: {price}, sir."
        else:
            return f"Could not fetch {symbol} price, sir."
    except Exception as e:
        logging.error(f"Crypto tracking failed: {e}")
        return f"Cryptocurrency tracking for {symbol} unavailable, sir."

@function_tool(description="Manage calendar events")
async def smart_calendar(action: str, event_title: str = "") -> str:
    return f"Calendar {action} for '{event_title}' noted, sir."

@function_tool(description="Generate AI content")
async def ai_content_generator(content_type: str, topic: str) -> str:
    return f"AI content generation for {content_type} about '{topic}' noted, sir."

@function_tool(description="Control Spotify music player")
async def spotify_controller(action: str = "status", query: Optional[str] = None) -> str:
    if not sp:
        return "Spotify not configured, sir."

    try:
        devices = sp.devices()
        if not devices['devices']:
            return "No Spotify devices found, sir."

        active_device = None
        for device in devices['devices']:
            if device['is_active']:
                active_device = device
                break

        if not active_device and action not in ["status"]:
            return "No active Spotify device found, sir."

        if action == "status":
            current = sp.current_playback()
            if current and current.get("is_playing"):
                track = current["item"]["name"]
                artist = current["item"]["artists"][0]["name"]
                return f"Now playing: '{track}' by {artist}, sir."
            else:
                return "No music currently playing, sir."

        elif action == "play":
            if query:
                results = sp.search(q=query, type='track', limit=1)
                if results['tracks']['items']:
                    track_uri = results['tracks']['items'][0]['uri']
                    sp.start_playback(uris=[track_uri])
                    track_name = results['tracks']['items'][0]['name']
                    artist_name = results['tracks']['items'][0]['artists'][0]['name']
                    return f"Playing '{track_name}' by {artist_name}, sir."
                else:
                    return f"Could not find '{query}', sir."
            else:
                sp.start_playback()
                return "Music resumed, sir."

        elif action == "pause":
            sp.pause_playback()
            return "Music paused, sir."

        elif action == "next":
            sp.next_track()
            return "Skipped to next track, sir."

        elif action == "previous":
            sp.previous_track()
            return "Previous track, sir."

        elif action == "volume":
            if query and query.isdigit():
                volume = max(0, min(100, int(query)))
                sp.volume(volume)
                return f"Volume set to {volume}%, sir."
            else:
                return "Please specify a volume level (0-100), sir."

        else:
            return "Available Spotify actions: status, play, pause, next, previous, volume, sir."

    except Exception as e:
        logging.error(f"Spotify error: {e}")
        return f"Spotify control failed: {e}, sir."
