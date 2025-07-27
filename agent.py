from dotenv import load_dotenv
import uuid
import logging
import asyncio
import datetime

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, ChatMessage
from livekit.plugins import noise_cancellation, google
from livekit.agents.llm.tool_context import function_tool # Ensure function_tool is imported here
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tools import (
    # Core tools (ensure these are all decorated with @function_tool in tools.py)
    get_weather,
    search_web,
    send_email,
    get_current_time,
    get_stock_price,
    get_news,
    system_stats,
    generate_qr_code,
    calculate,
    random_fact,
    # Advanced features (ensure these are all decorated with @function_tool in tools.py)
    create_mindmap,
    voice_memo_recorder,
    password_generator,
    data_visualizer,
    translate_text,
    habit_tracker,
    expense_tracker,
    file_organizer,
    screenshot_analyzer,
    network_scanner,
    code_snippet_manager,
    system_optimizer,
    crypto_tracker,
    smart_calendar,
    ai_content_generator,
    spotify_controller,
    # Memory tools (these are already decorated in tools.py, so import directly)
    memory_manager, # This is the class instance, not a tool itself
    log_interaction,
    get_conversation_history,
    store_user_data,
    retrieve_user_data,
    # Authentication tools (these are already decorated in tools.py)
    # We will import them as their decorated versions directly
    register_user_mock,
    login_user_mock,
)

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

class FridayAssistant(Agent):
    def __init__(self, initial_user_id: str, session_id: str) -> None:
        self._initial_user_id = initial_user_id
        self._current_user_id = initial_user_id
        self.session_id = session_id
        self.consecutive_errors = 0

        # Create dynamic instructions with context
        instructions = self._create_dynamic_instructions()

        super().__init__(
            instructions=instructions,
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",
                temperature=0.3,
                instructions=instructions,
            ),
            tools=[
                # Directly include all tools that are already decorated with @function_tool in tools.py
                get_weather,
                search_web,
                send_email,
                get_current_time,
                get_stock_price,
                get_news,
                system_stats,
                generate_qr_code,
                calculate,
                random_fact,
                create_mindmap,
                voice_memo_recorder,
                password_generator,
                data_visualizer,
                translate_text,
                habit_tracker,
                expense_tracker,
                file_organizer,
                screenshot_analyzer,
                network_scanner,
                code_snippet_manager,
                system_optimizer,
                crypto_tracker,
                smart_calendar,
                ai_content_generator,
                spotify_controller,
                log_interaction,
                get_conversation_history,
                store_user_data,
                retrieve_user_data,
                register_user_mock, # This is the actual tool from tools.py
                login_user_mock,    # This is the actual tool from tools.py
            ],
        )

    def _create_dynamic_instructions(self) -> str:
        """Create instructions with conversation history and user context"""
        try:
            recent_history = memory_manager.get_recent_history(self._current_user_id, self.session_id, limit=3)
            history_context = ""

            if recent_history:
                history_context = "\n\n# Recent Conversation Context (Current Session):\n"
                for entry in recent_history:
                    role = entry.get('role', 'unknown').capitalize()
                    content = entry.get('content', '')
                    if len(content) > 100:
                        content = content[:100] + "..."
                    history_context += f"- {role}: {content}\n"

            user_context = ""
            try:
                favorite_color = memory_manager.retrieve_user_data(self._current_user_id, "favorite_color")
                if favorite_color:
                    user_context = f"\n# User Preferences:\n- Favorite color: {favorite_color}\n"
            except Exception as e:
                logging.debug(f"Could not retrieve user context: {e}")

            return f"""
{AGENT_INSTRUCTION}

# Important Guidelines:
- You are Friday, an AI assistant. Be concise and helpful.
- Always end responses with "sir" when appropriate.
- Use tools when needed to fulfill requests.
- Do not repeat previous responses.
- Be conversational and remember user preferences when possible.

# Current User ID: {self._current_user_id}
# Current Session ID: {self.session_id}
{user_context}{history_context}

Remember: Be brief, helpful, and avoid repetition. Use tools appropriately and maintain user context.
"""
        except Exception as e:
            logging.error(f"Error creating dynamic instructions: {e}")
            return AGENT_INSTRUCTION

    async def on_message(self, message: ChatMessage):
        """Handle incoming messages and log them"""
        try:
            if message.text:
                logging.info(f"User message: {message.text}")
                # Log the user message using the directly imported log_interaction
                await log_interaction(self._current_user_id, self.session_id, "user", message.text)

            self.consecutive_errors = 0

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            self.consecutive_errors += 1

    async def on_agent_response(self, response: str):
        """Handle agent responses and log them"""
        try:
            if response:
                logging.info(f"Agent response: {response}")
                # Log the agent response using the directly imported log_interaction
                await log_interaction(self._current_user_id, self.session_id, "friday", response)
        except Exception as e:
            logging.error(f"Error logging agent response: {e}")

async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the agent"""
    session_id = str(uuid.uuid4())

    # Determine initial user_id from LiveKit client metadata or use anonymous
    initial_user_id = "anonymous_user_" + str(uuid.uuid4())[:8]
    
    try:
        # Try to get user ID from room metadata
        if hasattr(ctx.room, 'metadata') and ctx.room.metadata:
            if isinstance(ctx.room.metadata, dict) and 'userId' in ctx.room.metadata:
                initial_user_id = ctx.room.metadata['userId']
            elif isinstance(ctx.room.metadata, str):
                # Try to parse as JSON if it's a string
                import json
                try:
                    metadata = json.loads(ctx.room.metadata)
                    if 'userId' in metadata:
                        initial_user_id = metadata['userId']
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logging.debug(f"Could not extract user ID from metadata: {e}")

    if initial_user_id.startswith("anonymous_user_"):
        logging.warning(f"No explicit user ID provided by LiveKit client metadata. Starting with anonymous user_id: {initial_user_id}")

    logging.info(f"Starting new session for initial user: {initial_user_id}, session: {session_id}")

    try:
        # Retrieve recent interactions to provide context
        recent_interactions = memory_manager.get_recent_history(initial_user_id, session_id, limit=5)
        logging.info(f"Retrieved {len(recent_interactions)} recent interactions for user {initial_user_id}, session {session_id}")

        # Create agent instance
        friday_agent = FridayAssistant(initial_user_id, session_id)

        # Create session
        session = AgentSession()

        # Start session with proper configuration
        await session.start(
            room=ctx.room,
            agent=friday_agent,
            room_input_options=RoomInputOptions(
                video_enabled=True,
                audio_enabled=True,
                noise_cancellation=noise_cancellation.BVC(),
                close_on_disconnect=False,
            ),
        )

        # Connect to room
        await ctx.connect()

        # Log initial greeting using the directly imported log_interaction
        try:
            await log_interaction(initial_user_id, session_id, "friday", SESSION_INSTRUCTION)
        except Exception as e:
            logging.error(f"Failed to log initial greeting: {e}")

        # Keep session alive and handle any session-level events
        while True:
            try:
                await asyncio.sleep(1)
                
                # Optional: Add periodic cleanup or maintenance tasks here
                # For example, you could periodically check session health
                
            except KeyboardInterrupt:
                logging.info("Shutting down gracefully...")
                break
            except Exception as e:
                logging.error(f"Session error: {e}")
                await asyncio.sleep(5)

    except Exception as e:
        logging.error(f"Failed to start session: {e}")
        # Don't re-raise in production, let the system handle recovery
        raise

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
