import logging
import os
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Dict, List
import datetime
import json
from livekit.agents.llm.tool_context import function_tool
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
import datetime
import sqlite3 # Keep sqlite3 for other tools if they use it, but not for memory
import logging
from livekit.agents.llm.tool_context import function_tool
import random
import hashlib
import uuid
# Removed redundant sqlite3 import if it was already there
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
from langchain.agents import Tool
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from livekit.agents.llm.tool_context import function_tool
TEMP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_media")
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
from dotenv import load_dotenv
load_dotenv()

# --- Firebase Imports for Contextual Memory ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Global Firebase Initialization ---
db = None # Initialize db to None
try:
    firebase_credentials_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
    if not firebase_credentials_path or not os.path.exists(firebase_credentials_path):
        logging.error("FIREBASE_SERVICE_ACCOUNT_KEY_PATH environment variable not set or file not found.")
        # If Firebase is mandatory for your application, consider raising an exception here.
        # For now, we'll log an error and allow the app to run without memory features.
        raise FileNotFoundError("Firebase service account key file not found or path not set in .env.")

    cred = credentials.Certificate(firebase_credentials_path)
    if not firebase_admin._apps: # Initialize Firebase app only once
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase Firestore initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Firebase: {e}. Memory features will be unavailable.")
    db = None # Ensure db is None if initialization fails


# --- Global Spotify Configuration ---
SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

# Initialize SpotifyOAuth once globally, configured to use cache and not open browser
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SPOTIFY_SCOPE,
    open_browser=False,
    cache_path=".spotify_cache"
)
sp = spotipy.Spotify(auth_manager=sp_oauth)


# --- Contextual Memory Manager Class (Firestore Version) ---
class MemoryManager:
    def __init__(self, firestore_db):
        self.db = firestore_db
        if not self.db:
            logging.warning("Firestore DB not initialized. Memory features will be unavailable.")

    def log_interaction(self, session_id: str, role: str, content: str):
        if not self.db:
            logging.error("Cannot log interaction: Firestore DB not available.")
            return

        try:
            # Each session is a document, with messages as a subcollection
            session_ref = self.db.collection('sessions').document(session_id).collection('messages')
            session_ref.add({
                'role': role,
                'content': content,
                'timestamp': firestore.SERVER_TIMESTAMP # Use server timestamp for consistency
            })
            logging.info(f"Logged interaction for session {session_id}: {role} - {content[:50]}...")
        except Exception as e:
            logging.error(f"Error logging interaction to Firestore: {e}")

    def get_recent_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        if not self.db:
            logging.error("Cannot retrieve history: Firestore DB not available.")
            return []

        history = []
        try:
            messages_ref = self.db.collection('sessions').document(session_id).collection('messages')
            query = messages_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            docs = query.stream()
            for doc in docs:
                data = doc.to_dict()
                # Convert Firestore Timestamp object to string for easier use in prompt
                timestamp_str = data.get('timestamp').isoformat() if data.get('timestamp') else None
                history.append({
                    "role": data.get('role'),
                    "content": data.get('content'),
                    "timestamp": timestamp_str
                })
            # Reverse to get chronological order (oldest first)
            history.reverse()
            logging.info(f"Retrieved {len(history)} history items for session {session_id}.")
        except Exception as e:
            logging.error(f"Error retrieving history from Firestore: {e}")
        return history

# Instantiate the MemoryManager globally, passing the initialized Firestore client
# This ensures memory_manager is available to other tools and agent.
memory_manager = MemoryManager(db)


# --- New Tool for Logging Interactions (to be called by agent) ---
@function_tool(description="Log a conversation interaction for contextual memory. Takes 'session_id' (string), 'role' ('user' ('user') or 'friday' ('friday')), and 'content' (string).")
async def log_interaction(session_id: str, role: str, content: str) -> str:
    """
    Logs a conversation interaction to the contextual memory database.
    """
    try:
        memory_manager.log_interaction(session_id, role, content)
        return "Interaction logged successfully, sir."
    except Exception as e:
        logging.error(f"Failed to log interaction: {e}")
        return f"Failed to log interaction, sir: {e}"


# Original Enhanced Tools (rest of your tools.py content goes here)
# ... (all your existing tools like get_weather, search_web, send_email, etc.) ...
# Ensure these are still present in your file.

@function_tool(description="Get detailed weather information for a given city. Takes 'city' (string) as a required argument and 'units' (string, 'metric' or 'imperial') as an optional argument. Example: get_weather('London', units='metric')")
async def get_weather(
    city: str,
    units: str = "metric",
) -> str:
    """
    Get detailed weather information for a given city.

    Args:
        city: City name
        units: Temperature units ('metric' for Celsius, 'imperial' for Fahrenheit)
    """
    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=%l:+%c+%t+%h+%w+%p+%P",
            timeout=5
        )
        if response.status_code == 200:
            weather_data = response.text.strip()
            logging.info(f"Weather for {city}: {weather_data}")
            result_str = f"Weather report for {city}: {weather_data}"
            return result_str
        else:
            error_msg = f"Could not retrieve weather for {city}, sir. Status: {response.status_code}"
            logging.error(f"Failed to get weather for {city}: {response.status_code}")
            return error_msg
    except Exception as e:
        error_msg = f"Weather service seems to be having issues for {city}, sir. Error: {e}"
        logging.error(f"Error retrieving weather for {city}: {e}")
        return error_msg

# Corrected @tool decorator usage for LiveKit Agents
@function_tool(description="Search the web using DuckDuckGo to get current information. Input should be a concise search query.")
async def search_web(
    query: str,
) -> str:
    """
    Search the web using DuckDuckGo with enhanced results.

    Args:
        query: Search query
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(tool_input=query)
        logging.info(f"Search results for '{query}': {results}")
        return f"Search results for '{query}': {results}"
    except Exception as e:
        logging.error(f"Error searching the web for '{query}': {e}")
        return f"The web search encountered an issue, sir. Error: {e}"


@function_tool(description="Send an email through Gmail with enhanced features.")
async def send_email(
    to_email: str,
    subject: str,
    message: str,
    cc_email: Optional[str] = None,
    priority: str = "normal") -> str:
    """
    Send an email through Gmail with enhanced features.

    Args:
        to_email: Recipient email address
        subject: Email subject line
        message: Email body content
        cc_email: Optional CC email address
        priority: Email priority (low, normal, high)
    """
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        gmail_user = os.getenv("GMAIL_USER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")

        if not gmail_user or not gmail_password:
            logging.error("Gmail credentials not found")
            return "Email credentials not configured, sir."

        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg["Date"] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

        if priority.lower() == "high":
            msg["X-Priority"] = "1"
            msg["X-MSMail-Priority"] = "High"
            msg["X-MSMail-Priority"] = "High"
        elif priority.lower() == "low":
            msg["X-Priority"] = "5"
            msg["X-MSMail-Priority"] = "Low"

        recipients = [to_email]
        if cc_email:
            msg["Cc"] = cc_email
            recipients.append(cc_email)

        msg.attach(MIMEText(message, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, recipients, msg.as_string())

        logging.info(f"Email sent successfully to {to_email}")
        return f"Email dispatched to {to_email}, sir."

    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return f"Email sending encountered an issue, sir."

@function_tool(description="Get the current date and time with timezone support.")
async def get_current_time(
    timezone: str = "local") -> str:
    """
    Get the current date and time with timezone support.

    Args:
        timezone: Timezone (e.g., 'UTC', 'EST', 'PST', or 'local')
    """
    try:
        now = datetime.datetime.now()
        return f"Current time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"
    except Exception as e:
        logging.error(f"Error getting time: {e}")
        return "Time service unavailable, sir."

# 15 NEW ADVANCED FEATURES

@function_tool(description="Create a visual mind map with the central topic and branches.")
async def create_mindmap(
    central_topic: str,
    branches: List[str],
    filename: str = "mindmap.png") -> str:
    """
    Create a visual mind map with the central topic and branches.

    Args:
        central_topic: Main topic in the center
        branches: List of branch topics
        filename: Output filename
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axis('off')

        # Central topic
        central_circle = plt.Circle((0, 0), 2, color='lightblue', alpha=0.7)
        ax.add_patch(central_circle)
        ax.text(0, 0, central_topic, ha='center', va='center', fontsize=12, weight='bold')

        # Branches
        angles = np.linspace(0, 2*np.pi, len(branches), endpoint=False)
        for i, (branch, angle) in enumerate(zip(branches, angles)):
            x = 6 * np.cos(angle)
            y = 6 * np.sin(angle)

            # Draw line
            ax.plot([0, x], [0, y], 'k-', alpha=0.6)

            # Branch circle
            branch_circle = plt.Circle((x, y), 1.5, color='lightcoral', alpha=0.7)
            ax.add_patch(branch_circle)
            ax.text(x, y, branch, ha='center', va='center', fontsize=10)

        plt.title(f"Mind Map: {central_topic}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return f"Mind map created successfully: {filename}, sir."
    except Exception as e:
        logging.error(f"Error creating mind map: {e}")
        return "Mind map creation failed, sir."

@function_tool(description="Record a voice memo for specified duration.")
async def voice_memo_recorder(
    duration: int = 10,
    filename: str = "voice_memo.wav") -> str:
    """
    Record a voice memo for specified duration.

    Args:
        duration: Recording duration in seconds
        filename: Output filename
    """
    try:
        sample_rate = 44100
        logging.info(f"Recording voice memo for {duration} seconds...")

        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Save to file
        sf.write(filename, audio_data, sample_rate)

        return f"Voice memo recorded successfully: {filename}, sir."
    except Exception as e:
        logging.error(f"Error recording voice memo: {e}")
        return "Voice memo recording failed, sir."

@function_tool(description="Generate a secure password with specified criteria.")
async def password_generator(
    length: int = 16,
    include_symbols: bool = True,
    include_numbers: bool = True,
    exclude_ambiguous: bool = True) -> str:
    """
    Generate a secure password with specified criteria.

    Args:
        length: Password length
        include_symbols: Include special characters
        include_numbers: Include numbers
        exclude_ambiguous: Exclude ambiguous characters (0, O, l, I)
    """
    try:
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if include_numbers:
            chars += "0123456789"

        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        if exclude_ambiguous:
            chars = chars.replace('0', '').replace('O', '').replace('l', '').replace('I', '')

        password = ''.join(random.choice(chars) for _ in range(length))

        # Calculate strength
        strength = "Weak"
        if length >= 12 and include_numbers and include_symbols:
            strength = "Very Strong"
        elif length >= 8 and (include_numbers or include_symbols):
            strength = "Strong"
        elif length >= 8:
            strength = "Medium"

        return f"Generated password: {password} (Strength: {strength}), sir."
    except Exception as e:
        logging.error(f"Error generating password: {e}")
        return "Password generation failed, sir."

@function_tool(description="Create data visualizations from provided data points.")
async def data_visualizer(
    data_points: List[Dict],
    chart_type: str = "bar",
    title: str = "Data Visualization") -> str:
    """
    Create data visualizations from provided data points.

    Args:
        data_points: List of dictionaries with x and y values
        chart_type: Type of chart (bar, line, pie, scatter)
        title: Chart title
    """
    try:
        df = pd.DataFrame(data_points)

        plt.figure(figsize=(10, 6))

        if chart_type == "bar":
            plt.bar(df['x'], df['y'])
        elif chart_type == "line":
            plt.plot(df['x'], df['y'], marker='o')
        elif chart_type == "pie":
            plt.pie(df['y'], labels=df['x'], autopct='%1.1f%%')
        elif chart_type == "scatter":
            plt.scatter(df['x'], df['y'])

        plt.title(title)
        plt.xlabel('X Values')
        plt.ylabel('Y Values')

        filename = f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return f"Data visualization created: {filename}, sir."
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
        return "Data visualization failed, sir."


@function_tool(description="Translate text between languages. Takes 'text' (string) to translate, 'target_language' (string, e.g., 'en', 'es', 'fr', 'de') as an optional argument (defaults to 'en'), and 'source_language' (string, e.g., 'auto' for auto-detection, or specific code like 'en', 'es') as an optional argument. Example: translate_text('Hello', target_language='es') or translate_text('Bonjour', target_language='en', source_language='fr')")
async def translate_text(
    text: str,
    target_language: str = "en",
    source_language: str = "auto",
) -> str:
    """
    Translate text between languages.

    Args:
        text: Text to translate
        target_language: Target language code (en, es, fr, de, etc.)
        source_language: Source language code (auto for detection)
    """
    try:
        translator = GoogleTranslator(source=source_language, target=target_language)
        translated_text = translator.translate(text)
        result_message = f"Translation ({source_language} → {target_language}): {translated_text}, sir."
        return result_message
    except Exception as e:
        error_msg = f"Translation service unavailable, sir. Error: {e}"
        logging.error(f"Error translating text: {e}")
        return error_msg

@function_tool(description="Track daily habits and progress.")
async def habit_tracker(
    habit_name: str,
    action: str = "check",
    target_days: int = 30) -> str:
    """
    Track daily habits and progress.

    Args:
        habit_name: Name of the habit
        action: Action to perform (check, status, reset)
        target_days: Target number of days
    """
    try:
        db_path = "habits.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                streak INTEGER DEFAULT 0,
                last_check DATE,
                target_days INTEGER,
                created_date DATE
            )
        ''')

        if action == "check":
            today = datetime.date.today()
            cursor.execute('''
                INSERT OR REPLACE INTO habits (name, streak, last_check, target_days, created_date)
                VALUES (?,
                    CASE
                        WHEN (SELECT last_check FROM habits WHERE name = ?) = ?
                        THEN (SELECT streak FROM habits WHERE name = ?)
                        WHEN (SELECT last_check FROM habits WHERE name = ?) = date(?, '-1 day')
                        THEN (SELECT streak FROM habits WHERE name = ?) + 1
                        ELSE 1
                    END,
                    ?, ?, COALESCE((SELECT created_date FROM habits WHERE name = ?), ?))
            ''', (habit_name, habit_name, today, habit_name, habit_name, today, habit_name, today, target_days, habit_name, today))

            cursor.execute('SELECT streak FROM habits WHERE name = ?', (habit_name,))
            streak = cursor.fetchone()[0]

            conn.commit()
            conn.close()

            progress = (streak / target_days) * 100
            return f"Habit '{habit_name}' checked! Current streak: {streak} days ({progress:.1f}% of target), sir."

        elif action == "status":
            cursor.execute('SELECT streak, target_days, created_date FROM habits WHERE name = ?', (habit_name,))
            result = cursor.fetchone()

            if result:
                streak, target, created = result
                progress = (streak / target) * 100
                return f"Habit '{habit_name}': {streak}/{target} days ({progress:.1f}% complete), sir."
            else:
                return f"Habit '{habit_name}' not found, sir."

        conn.close()

    except Exception as e:
        logging.error(f"Error with habit tracker: {e}")
        return "Habit tracking system unavailable, sir."

@function_tool(description="Track personal expenses and generate reports.")
async def expense_tracker(
    amount: float,
    category: str,
    description: str = "",
    action: str = "add") -> str:
    """
    Track personal expenses and generate reports.

    Args:
        amount: Expense amount
        category: Expense category
        description: Optional description
        action: Action to perform (add, report, categories)
    """
    try:
        db_path = "expenses.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY,
                amount REAL,
                category TEXT,
                description TEXT,
                date DATE,
                created_at TIMESTAMP
            )
        ''')

        if action == "add":
            today = datetime.date.today()
            cursor.execute('''
                INSERT INTO expenses (amount, category, description, date, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (amount, category, description, today, datetime.datetime.now()))

            conn.commit()
            conn.close()

            return f"Expense added: ${amount:.2f} for {category}, sir."

        elif action == "report":
            cursor.execute('''
                SELECT category, SUM(amount) as total
                FROM expenses
                WHERE date >= date('now', '-30 days')
                GROUP BY category
                ORDER BY total DESC
            ''')

            results = cursor.fetchall()

            if results:
                report = "Monthly expense report:\n"
                total = sum(row[1] for row in results)
                for category, cat_total in results:
                    percentage = (cat_total / total) * 100
                    report += f"{category}: ${cat_total:.2f} ({percentage:.1f}%)\n"
                report += f"Total: ${total:.2f}"

                conn.close()
                return report + ", sir."
            else:
                return "No expenses recorded in the last 30 days, sir."

        conn.close()

    except Exception as e:
        logging.error(f"Error with expense tracker: {e}")
        return "Expense tracking system unavailable, sir."

@function_tool(description="Organize files in a directory by type, date, or size.")
async def file_organizer(
    directory_path: str,
    organize_by: str = "extension") -> str:
    """
    Organize files in a directory by type, date, or size.

    Args:
        directory_path: Path to directory to organize
        organize_by: Organization method (extension, date, size)
    """
    try:
        if not os.path.exists(directory_path):
            return f"Directory not found: {directory_path}, sir."

        organized_count = 0

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path):
                if organize_by == "extension":
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower() if ext else "no_extension"
                    folder_name = ext[1:] if ext.startswith('.') else ext

                elif organize_by == "date":
                    mtime = os.path.getmtime(file_path)
                    date = datetime.datetime.fromtimestamp(mtime)
                    folder_name = date.strftime("%Y-%m")

                elif organize_by == "size":
                    size = os.path.getsize(file_path)
                    if size < 1024 * 1024:  # < 1MB
                        folder_name = "small_files"
                    elif size < 10 * 1024 * 1024:  # < 10MB
                        folder_name = "medium_files"
                    else:
                        folder_name = "large_files"

                # Create folder if it doesn't exist
                folder_path = os.path.join(directory_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Move file
                new_path = os.path.join(folder_path, filename)
                shutil.move(file_path, new_path)
                organized_count += 1

        return f"Organized {organized_count} files by {organize_by}, sir."

    except Exception as e:
        logging.error(f"Error organizing files: {e}")
        return "File organization failed, sir."

@function_tool(description="Take and analyze screenshots for productivity insights.")
async def screenshot_analyzer(
    analysis_type: str = "basic") -> str:
    """
    Take and analyze screenshots for productivity insights.

    Args:
        analysis_type: Type of analysis (basic, text, colors)
    """
    try:
        # Take screenshot
        screenshot = pyautogui.screenshot()
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot.save(filename)

        if analysis_type == "basic":
            width, height = screenshot.size
            return f"Screenshot captured: {filename} ({width}x{height}), sir."

        elif analysis_type == "colors":
            # Analyze dominant colors
            pixels = np.array(screenshot)
            pixels = pixels.reshape(-1, 3)

            # Get dominant colors (simplified)
            unique_colors = np.unique(pixels, axis=0)

            return f"Screenshot analyzed: {filename}. Found {len(unique_colors)} unique colors, sir."

    except Exception as e:
        logging.error(f"Error with screenshot analysis: {e}")
        return "Screenshot analysis failed, sir."

@function_tool(description="Scan network for devices and security analysis.")
async def network_scanner(
    scan_type: str = "ports") -> str:
    """
    Scan network for devices and security analysis.

    Args:
        scan_type: Type of scan (ports, devices, speed)
    """
    try:
        if scan_type == "ports":
            # Scan common ports on localhost
            common_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 8080]
            open_ports = []

            for port in common_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1)
                        result = s.connect_ex(('127.0.0.1', port))
                        if result == 0:
                            open_ports.append(port)
                except:
                    pass

            if open_ports:
                return f"Open ports detected: {', '.join(map(str, open_ports))}, sir."
            else:
                return "No common ports found open, sir."

        elif scan_type == "speed":
            # Network speed test
            st = speedtest.Speedtest()
            st.get_best_server()

            download_speed = st.download() / 1_000_000
            upload_speed = st.upload() / 1_000_200
            ping = st.results.ping

            return f"Network speed: {download_speed:.2f} Mbps down, {upload_speed:.2f} Mbps up, {ping:.2f}ms ping, sir."

    except Exception as e:
        logging.error(f"Error with network scanner: {e}")
        return "Network scanning failed, sir."

@function_tool(description="Manage and store code snippets with tags and search.")
async def code_snippet_manager(
    action: str,
    language: str = "python",
    snippet_name: str = "",
    code: str = "") -> str:
    """
    Manage and store code snippets with tags and search.

    Args:
        action: Action to perform (save, search, list, get)
        language: Programming language
        snippet_name: Name for the snippet
        code: Code content
    """
    try:
        db_path = "code_snippets.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY,
                name TEXT,
                language TEXT,
                code TEXT,
                created_at TIMESTAMP,
                tags TEXT
            )
        ''')

        if action == "save":
            cursor.execute('''
                INSERT INTO snippets (name, language, code, created_at, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (snippet_name, language, code, datetime.datetime.now(), ""))

            conn.commit()
            conn.close()

            return f"Code snippet '{snippet_name}' saved for {language}, sir."

        elif action == "list":
            cursor.execute('SELECT name, language FROM snippets ORDER BY created_at DESC')
            results = cursor.fetchall()

            if results:
                snippet_list = "Available code snippets:\n"
                for name, lang in results:
                    snippet_list += f"- {name} ({lang})\n"

                conn.close()
                return snippet_list + "sir."
            else:
                return "No code snippets found, sir."

        conn.close()

    except Exception as e:
        logging.error(f"Error with code snippet manager: {e}")
        return "Code snippet management failed, sir."

@function_tool(description="Optimize system performance and clean up resources.")
async def system_optimizer(
    optimization_type: str = "cleanup") -> str:
    """
    Optimize system performance and clean up resources.

    Args:
        optimization_type: Type of optimization (cleanup, memory, disk)
    """
    try:
        if optimization_type == "cleanup":
            # Clean temporary files
            temp_dir = tempfile.gettempdir()
            temp_files = os.listdir(temp_dir)
            cleaned_files = 0

            for file in temp_files:
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleaned_files += 1
                except:
                    pass

            return f"System cleanup completed. Removed {cleaned_files} temporary files, sir."

        elif optimization_type == "memory":
            # Get memory usage
            memory = psutil.virtual_memory()

            # Force garbage collection
            import gc
            gc.collect()

            return f"Memory optimization completed. Available: {memory.available // (1024**3)}GB, sir."

        elif optimization_type == "disk":
            # Get disk usage
            disk = psutil.disk_usage('/')
            free_gb = disk.free // (1024**3)

            return f"Disk analysis completed. Free space: {free_gb}GB, sir."

    except Exception as e:
        logging.error(f"Error with system optimization: {e}")
        return "System optimization failed, sir."

@function_tool(description="Track cryptocurrency prices and market data.")
async def crypto_tracker(
    symbol: str = "BTC",
    action: str = "price") -> str:
    """
    Track cryptocurrency prices and market data.

    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, etc.)
        action: Action to perform (price, trend, portfolio)
    """
    try:
        if action == "price":
            # Using a free crypto API
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd&include_24hr_change=true"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if symbol.lower() in data:
                    price = data[symbol.lower()]['usd']
                    change = data[symbol.lower()].get('usd_24h_change', 0)

                    return f"{symbol.upper()}: ${price:,.2f} (24h: {change:+.2f}%), sir."
                else:
                    return f"Cryptocurrency {symbol} not found, sir."
            else:
                return "Cryptocurrency data unavailable, sir."

    except Exception as e:
        logging.error(f"Error with crypto tracker: {e}")
        return "Cryptocurrency tracking failed, sir." 

@function_tool(description="Manage calendar events with smart scheduling. Actions include 'add' (requires event_title, event_date YYYY-MM-DD, event_time HH:MM), 'list' (lists events for a specific date, requires event_date YYYY-MM-DD), and 'upcoming' (lists up to 5 upcoming events).")
async def smart_calendar(
    action: str,
    event_title: str = "",
    event_date: str = "", # This will be used for specific date listing
    event_time: str = "") -> str:
    """
    Manage calendar events with smart scheduling.

    Args:
        action: Action to perform (add, list, upcoming)
        event_title: Title of the event (for 'add' action)
        event_date: Date in YYYY-MM-DD format (for 'add' and 'list' actions)
        event_time: Time in HH:MM format (for 'add' action)
    """
    db_path = "calendar.db"
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                title TEXT,
                date TEXT, -- Changed to TEXT to avoid SQLite's implicit conversions
                time TEXT, -- Changed to TEXT
                created_at TIMESTAMP
            )
        ''')

        if action == "add":
            if not event_title or not event_date or not event_time:
                return "Please provide a title, date (YYYY-MM-DD), and time (HH:MM) to add an event, sir."

            try:
                # Validate and normalize date/time to ensure consistent format
                parsed_date = datetime.datetime.strptime(event_date, "%Y-%m-%d").strftime("%Y-%m-%d")
                parsed_time = datetime.datetime.strptime(event_time, "%H:%M").strftime("%H:%M")
            except ValueError:
                return "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time, sir."

            cursor.execute('''
                INSERT INTO events (title, date, time, created_at)
                VALUES (?, ?, ?, ?)
            ''', (event_title, parsed_date, parsed_time, datetime.datetime.now()))

            conn.commit() # Commit changes immediately after insertion
            return f"Event '{event_title}' added for {parsed_date} at {parsed_time}, sir."

        elif action == "list":
            if not event_date:
                return "Please provide a date (YYYY-MM-DD) to list events, sir."

            try:
                # Validate and normalize date for querying
                parsed_date = datetime.datetime.strptime(event_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                return "Invalid date format. Please use YYYY-MM-DD for date, sir."

            cursor.execute('''
                SELECT title, date, time
                FROM events
                WHERE date = ?
                ORDER BY time
            ''', (parsed_date,)) # Use the parsed_date for consistency

            results = cursor.fetchall()

            if results:
                events_list = f"Events on {parsed_date}:\n"
                for title, date, time in results:
                    events_list += f"- {title}: {time}\n"
                return events_list + "sir."
            else:
                return f"No events found on {parsed_date}, sir."

        elif action == "upcoming":
            cursor.execute('''
                SELECT title, date, time
                FROM events
                WHERE date >= date('now')
                ORDER BY date, time
                LIMIT 5
            ''')

            results = cursor.fetchall()

            if results:
                events_list = "Upcoming events:\n"
                for title, date, time in results:
                    events_list += f"- {title}: {date} at {time}\n"

                return events_list + "sir."
            else:
                return "No upcoming events, sir."

        else:
            return "Invalid action for calendar. Please use 'add', 'list', or 'upcoming', sir."

    except Exception as e:
        logging.error(f"Error with smart calendar: {e}")
        return "Calendar management failed, sir."
    finally:
        if conn:
            conn.close() # Ensure the connection is always closed    
            
@function_tool(description="Generate various types of content using AI assistance.")
async def ai_content_generator(
    content_type: str,
    topic: str,
    style: str = "professional") -> str:
    """
    Generate various types of content using AI assistance.

    Args:
        content_type: Type of content (email, letter, summary, outline)
        topic: Topic or subject matter
        style: Writing style (professional, casual, creative)
    """
    try:
        templates = {
            "email": {
                "professional": "Subject: {topic}\n\nDear [Recipient],\n\nI hope this email finds you well. I am writing to discuss {topic}...\n\nBest regards,\n[Your Name]",
                "casual": "Hey!\n\nHope you're doing great! Just wanted to chat about {topic}...\n\nTalk soon!\n[Your Name]"
            },
            "outline": {
                "professional": "I. Introduction to {topic}\nII. Key Points\n   A. Main concept\n   B. Supporting details\nIII. Analysis\nIV. Conclusion",
                "creative": "🎯 {topic} Overview\n💡 Big Ideas\n🔍 Deep Dive\n✨ Creative Applications\n🎉 Wrap-up"
            }
        }

        if content_type in templates and style in templates[content_type]:
            content = templates[content_type][style].format(topic=topic)
            return f"Generated {content_type} content:\n\n{content}\n\n---\nContent generated for: {topic}, sir."
        else:
            return f"Content generation for {content_type} in {style} style is not available, sir."

    except Exception as e:
        logging.error(f"Error with AI content generator: {e}")
        return "Content generation failed, sir."

# Additional utility functions
@function_tool(description="Get stock price information.")
async def get_stock_price(
    symbol: str,
    period: str = "1d") -> str:
    """
    Get stock price information.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty:
            return f"Stock symbol {symbol} not found, sir."

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        return f"{symbol.upper()}: ${current_price:.2f} ({change:+.2f}, {change_percent:+.2f}%)"
    except Exception as e:
        logging.error(f"Error getting stock price for {symbol}: {e}")
        return f"Stock data unavailable for {symbol}, sir."

@function_tool(description="Get latest news headlines.")
async def get_news(
    query: str = "latest news",
    country: str = "us") -> str:
    """
    Get latest news headlines.

    Args:
        query: News search query
        country: Country code for news (us, uk, ca, etc.)
    """
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return "News service not configured, sir."

        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}"
        if query != "latest news":
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&sortBy=publishedAt"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])[:3]

                    if not articles:
                        return f"No news found for '{query}', sir."

                    news_summary = f"Top news for '{query}':\n"
                    for i, article in enumerate(articles, 1):
                        news_summary += f"{i}. {article['title']}\n"

                    return news_summary
                else:
                    return "News service unavailable, sir."
    except Exception as e:
        logging.error(f"Error getting news: {e}")
        return "News service encountered an issue, sir."

@function_tool(description="Get system performance statistics.")
async def system_stats() -> str:
    """
    Get system performance statistics.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        stats = f"System Status:\n"
        stats += f"CPU: {cpu_percent}%\n"
        stats += f"Memory: {memory.percent}% ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)\n"
        stats += f"Disk: {disk.percent}% ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)"

        return stats
    except Exception as e:
        logging.error(f"Error getting system stats: {e}")
        return "System monitoring unavailable, sir."

@function_tool(description="Generate a QR code for given data.")
async def generate_qr_code(
    data: str,
    size: int = 10) -> str:
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        filename = os.path.join(TEMP_OUTPUT_DIR, f"qr_{hashlib.md5(data.encode()).hexdigest()[:6]}.png")
        img.save(filename)

        return f"QR code generated and saved as: {filename}, sir."
    except Exception as e:
        logging.error(f"Error generating QR code: {e}")
        return "QR code generation failed, sir."

@function_tool(description="Perform mathematical calculations.")
async def calculate(
    expression: str,
) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Invalid calculation expression, sir."

        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        logging.error(f"Error calculating {expression}: {e}")
        return "Calculation failed, sir."

@function_tool(description="Get a random interesting fact.")
async def random_fact() -> str:
    """
    Get a random interesting fact.
    """
    try:
        facts = [
            "Octopuses have three hearts and blue blood.",
            "A group of flamingos is called a 'flamboyance'.",
            "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
            "The human brain has about 86 billion neurons.",
            "A single cloud can weigh more than a million pounds.",
            "The shortest war in history lasted only 38-45 minutes.",
            "There are more possible games of chess than atoms in the observable universe.",
            "Bananas are berries, but strawberries aren't.",
            "The Great Wall of China isn't visible from space with the naked eye.",
            "A jiffy is an actual unit of time - 1/100th of a second."
        ]

        fact = random.choice(facts)
        return f"Random fact: {fact}"
    except Exception as e:
        logging.error(f"Error getting random fact: {e}")
        return "Fact database unavailable, sir."


@function_tool(description="Control your Spotify music player. Actions: play, pause, next, previous, shuffle, repeat, status. For 'play' action, 'query' (song name) is optional to play a specific song, otherwise it resumes. Shuffle can be True/False. Repeat can be 'track', 'context', or 'off'.")
async def spotify_controller(
    action: str = "status",
    query: Optional[str] = None, # Added query for song search
    shuffle: Optional[bool] = None, # Changed to Optional to handle default behavior better
    repeat: Optional[str] = None  # Changed to Optional
) -> str:
    """
    Control Spotify playback.

    Args:
        action: Action to perform (play, pause, next, previous, status)
        query: Optional song name to play (for 'play' action)
        shuffle: Enable shuffle (True/False)
        repeat: Repeat mode (off/context/track)
    """
    try:
        # Get active device
        devices = sp.devices()
        active_device_id = None
        for device in devices['devices']:
            if device['is_active']:
                active_device_id = device['id']
                break

        if not active_device_id:
            return "No active Spotify device found, sir. Please open and activate Spotify on a device."

        # Apply shuffle state if provided
        if shuffle is not None:
            sp.shuffle(state=shuffle, device_id=active_device_id)

        # Apply repeat state if provided and valid
        if repeat in ["track", "context", "off"]:
            sp.repeat(repeat, device_id=active_device_id)

        if action == "play":
            if query:
                # Search for the track
                results = sp.search(q=query, type='track', limit=1)
                tracks = results['tracks']['items']
                if not tracks:
                    return f"Could not find any track named '{query}', sir."

                track_uri = tracks[0]['uri']
                track_name = tracks[0]['name']
                artist_name = tracks[0]['artists'][0]['name']
                sp.start_playback(device_id=active_device_id, uris=[track_uri])
                return f"Playing '{track_name}' by {artist_name}, sir."
            else:
                # Resume playback
                current_playback = sp.current_playback()
                if current_playback and current_playback.get('actions', {}).get('disallows', {}).get('resuming'):
                    return "Resuming playback is currently disallowed for the active track, sir. Please try playing a new song."
                sp.start_playback(device_id=active_device_id)
                return "Music resumed, sir."

        elif action == "pause":
            sp.pause_playback(device_id=active_device_id)
            return "Music paused, sir."
        elif action == "next":
            sp.next_track(device_id=active_device_id)
            return "Skipped to the next track, sir."
        elif action == "previous":
            sp.previous_track(device_id=active_device_id)
            return "Returned to the previous track, sir."
        elif action == "status":
            current = sp.current_playback()
            if not current or not current.get("is_playing"):
                return "No music is currently playing, sir."
            item = current["item"]
            track_name = item["name"]
            artists = ", ".join([artist["name"] for artist in item["artists"]])
            progress_ms = current.get("progress_ms", 0)
            duration_ms = item.get("duration_ms", 1)
            percent = (progress_ms / duration_ms) * 100
            return f"Currently playing: '{track_name}' by {artists} ({percent:.1f}% complete), sir."
        else:
            return "Invalid action for Spotify, sir."

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 403:
            return f"Spotify error: Permission denied or restriction violated, sir. This might be due to Spotify Free tier limitations or specific playback rules. Error details: {str(e)}"
        elif e.http_status == 404:
            return f"Spotify error: No active device found or device not available, sir. Please ensure your Spotify client is active and playing on a device. Error details: {str(e)}"
        return f"Spotify API error: {str(e)}, sir."
    except Exception as e:
        return f"Spotify control failed, sir. A system anomaly occurred: {e}"
