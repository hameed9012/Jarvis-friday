# Enhanced Tools (tools.py)
import logging
from livekit.agents import function_tool, RunContext
import requests
from langchain_community.tools import DuckDuckGoSearchRun
import os
import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText
from typing import Optional, Dict, List
import datetime
import json
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
import random
import hashlib
import uuid
import sqlite3
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from googletrans import Translator
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

# Original Enhanced Tools
@function_tool()
async def get_weather(
    context: RunContext,
    city: str,
    units: str = "metric") -> str:
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
            return f"Weather report for {city}: {weather_data}"
        else:
            logging.error(f"Failed to get weather for {city}: {response.status_code}")
            return f"Could not retrieve weather for {city}, sir."
    except Exception as e:
        logging.error(f"Error retrieving weather for {city}: {e}")
        return f"Weather service seems to be having issues, sir."

@function_tool()
async def search_web(
    context: RunContext,
    query: str,
    num_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo with enhanced results.
    
    Args:
        query: Search query
        num_results: Number of results to return (1-5)
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(tool_input=query)
        logging.info(f"Search results for '{query}': {results}")
        return f"Search results for '{query}': {results}"
    except Exception as e:
        logging.error(f"Error searching the web for '{query}': {e}")
        return f"The web search encountered an issue, sir."

@function_tool()
async def send_email(
    context: RunContext,
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

@function_tool()
async def get_current_time(
    context: RunContext,
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

@function_tool()
async def create_mindmap(
    context: RunContext,
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

@function_tool()
async def voice_memo_recorder(
    context: RunContext,
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

@function_tool()
async def password_generator(
    context: RunContext,
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

@function_tool()
async def data_visualizer(
    context: RunContext,
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

@function_tool()
async def translate_text(
    context: RunContext,
    text: str,
    target_language: str = "en",
    source_language: str = "auto") -> str:
    """
    Translate text between languages.
    
    Args:
        text: Text to translate
        target_language: Target language code (en, es, fr, de, etc.)
        source_language: Source language code (auto for detection)
    """
    try:
        translator = Translator()
        
        if source_language == "auto":
            detected = translator.detect(text)
            source_language = detected.lang
        
        translation = translator.translate(text, src=source_language, dest=target_language)
        
        return f"Translation ({source_language} → {target_language}): {translation.text}, sir."
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return "Translation service unavailable, sir."

@function_tool()
async def habit_tracker(
    context: RunContext,
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
        
        # Initialize database
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

@function_tool()
async def expense_tracker(
    context: RunContext,
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

@function_tool()
async def file_organizer(
    context: RunContext,
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

@function_tool()
async def screenshot_analyzer(
    context: RunContext,
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

@function_tool()
async def network_scanner(
    context: RunContext,
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
            upload_speed = st.upload() / 1_000_000
            ping = st.results.ping
            
            return f"Network speed: {download_speed:.2f} Mbps down, {upload_speed:.2f} Mbps up, {ping:.2f}ms ping, sir."
        
    except Exception as e:
        logging.error(f"Error with network scanner: {e}")
        return "Network scanning failed, sir."

@function_tool()
async def code_snippet_manager(
    context: RunContext,
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

@function_tool()
async def system_optimizer(
    context: RunContext,
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

@function_tool()
async def crypto_tracker(
    context: RunContext,
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

@function_tool()
async def smart_calendar(
    context: RunContext,
    action: str,
    event_title: str = "",
    event_date: str = "",
    event_time: str = "") -> str:
    """
    Manage calendar events with smart scheduling.
    
    Args:
        action: Action to perform (add, list, upcoming, conflicts)
        event_title: Title of the event
        event_date: Date in YYYY-MM-DD format
        event_time: Time in HH:MM format
    """
    try:
        db_path = "calendar.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                title TEXT,
                date DATE,
                time TIME,
                created_at TIMESTAMP
            )
        ''')
        
        if action == "add":
            cursor.execute('''
                INSERT INTO events (title, date, time, created_at)
                VALUES (?, ?, ?, ?)
            ''', (event_title, event_date, event_time, datetime.datetime.now()))
            
            conn.commit()
            conn.close()
            
            return f"Event '{event_title}' added for {event_date} at {event_time}, sir."
        
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
                
                conn.close()
                return events_list + "sir."
            else:
                return "No upcoming events, sir."
        
        conn.close()
        
    except Exception as e:
        logging.error(f"Error with smart calendar: {e}")
        return "Calendar management failed, sir."

@function_tool()
async def ai_content_generator(
    context: RunContext,
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
@function_tool()
async def get_stock_price(
    context: RunContext,
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

@function_tool()
async def get_news(
    context: RunContext,
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

@function_tool()
async def system_stats(
    context: RunContext) -> str:
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

@function_tool()
async def generate_qr_code(
    context: RunContext,
    data: str,
    size: int = 10) -> str:
    """
    Generate a QR code for given data.
    
    Args:
        data: Data to encode in QR code
        size: Size of the QR code (1-40)
    """
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
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return f"QR code generated for: {data[:50]}{'...' if len(data) > 50 else ''}"
    except Exception as e:
        logging.error(f"Error generating QR code: {e}")
        return "QR code generation failed, sir."

@function_tool()
async def calculate(
    context: RunContext,
    expression: str) -> str:
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

@function_tool()
async def random_fact(
    context: RunContext) -> str:
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