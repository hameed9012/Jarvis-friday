from dotenv import load_dotenv
import uuid # Import uuid for unique session IDs
import logging # Import logging for better debugging
import asyncio # Import asyncio for create_task

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    noise_cancellation,
)
from livekit.plugins import google
# Import AGENT_INSTRUCTION and SESSION_INSTRUCTION from prompts, but we'll modify SESSION_INSTRUCTION dynamically
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION 
from tools import (
    # Core tools
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

    # 15 NEW ADVANCED FEATURES
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

    # 🎧 New Tool
    spotify_controller,

    # --- New Tools for Contextual Memory ---
    memory_manager, # Import the global instance of MemoryManager
    log_interaction, # Import the tool to log interactions
)

load_dotenv()

# --- Enhanced AGENT_INSTRUCTION for Emotional Intelligence ---
# We will dynamically prepend conversation history to this instruction.
# The instruction itself will contain cues for emotional intelligence.
EMOTIONAL_INTELLIGENCE_INSTRUCTION = """
# Emotional Intelligence & Contextual Awareness
- **Analyze User Sentiment:** Pay close attention to the user's tone and emotional state (e.g., frustration, excitement, confusion, gratitude).
- **Adapt Tone:** If the user expresses frustration or sadness, respond with a more empathetic, calm, and supportive tone. If they are excited or happy, match their enthusiasm. For confusion, be more patient and clear.
- **Acknowledge Context:** Refer to previous turns in the conversation when relevant to demonstrate memory and continuity. Use phrases like "As we discussed earlier, sir...", "Regarding your previous query about...", or "Continuing from our last interaction..."
- **Personalize Responses:** Use information from past interactions (if available in context) to tailor your responses.
- **Maintain Persona:** While adapting tone, always maintain your core Friday persona (sophisticated, witty, efficient butler).

# Recent Conversation History (for context):
{conversation_history}

""" + AGENT_INSTRUCTION # Append the original AGENT_INSTRUCTION


class FridayAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=EMOTIONAL_INTELLIGENCE_INSTRUCTION, # Use the enhanced instruction
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",
                temperature=0.8,
            ),
            tools=[
                # Core functionality
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

                # 15 Advanced Features
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

                # 🎧 Spotify DJ mode
                spotify_controller,

                # --- New Tool for Contextual Memory ---
                log_interaction,
            ],
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession()
    # Generate a unique session ID for this conversation
    session_id = str(uuid.uuid4())
    logging.info(f"New session started with ID: {session_id}")

    await session.start(
        room=ctx.room,
        agent=FridayAssistant(),
        room_input_options=RoomInputOptions(
            # Enhanced input options
            video_enabled=True,
            audio_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # --- Dynamic Instruction Generation for Contextual Memory ---
    # Fetch recent history for the initial prompt
    recent_history = memory_manager.get_recent_history(session_id=session_id, limit=5)
    history_string = ""
    if recent_history:
        history_string = "Here's a summary of our recent conversation:\n"
        for entry in recent_history:
            # Format timestamp for readability, e.g., "HH:MM:SS"
            formatted_timestamp = datetime.datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S') if entry['timestamp'] else "N/A"
            history_string += f"- {entry['role'].capitalize()} ({formatted_timestamp}): {entry['content']}\n"
    else:
        history_string = "No prior conversation history for this session.\n"

    # Combine dynamic history with the static instruction
    # The `EMOTIONAL_INTELLIGENCE_INSTRUCTION` already has the {conversation_history} placeholder
    # and appends the original AGENT_INSTRUCTION.
    final_instructions = EMOTIONAL_INTELLIGENCE_INSTRUCTION.format(conversation_history=history_string)

    # Start the enhanced conversation
    await session.generate_reply(
        instructions=final_instructions,
        # Add a callback to log user input and Friday's response
        # This will be called after each turn
        on_reply=lambda user_input, friday_response: asyncio.create_task(
            log_interaction(session_id, "user", user_input)
        ) and asyncio.create_task(
            log_interaction(session_id, "friday", friday_response)
        )
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

