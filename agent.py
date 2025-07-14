from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    noise_cancellation,
)
from livekit.plugins import google
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
)

load_dotenv()

class FridayAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
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
                create_mindmap,          # Visual mind mapping
                voice_memo_recorder,     # Voice recording and transcription
                password_generator,      # Secure password generation
                data_visualizer,         # Data visualization and charts
                translate_text,          # Multi-language translation
                habit_tracker,           # Personal habit tracking
                expense_tracker,         # Financial expense management
                file_organizer,          # Intelligent file organization
                screenshot_analyzer,     # Screen capture and analysis
                network_scanner,         # Network security scanning
                code_snippet_manager,    # Code snippet storage and management
                system_optimizer,        # System performance optimization
                crypto_tracker,          # Cryptocurrency price tracking
                smart_calendar,          # Advanced calendar management
                ai_content_generator,    # AI-powered content creation
            ],
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession()

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

    # Start the enhanced conversation
    await session.generate_reply(
        instructions=SESSION_INSTRUCTION,
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))