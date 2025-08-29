# import asyncio
# import logging
# from time import perf_counter

# from livekit import rtc, api
# from livekit.agents import (
#     AgentSession,
#     JobContext,
#     WorkerOptions,
#     cli,
#     JobProcess
# )
# from livekit.plugins import silero

# # Import custom components
# from custom_llm import CustomLLM
# from custom_asr import CustomASR
# from custom_tts import CustomTTS
# from agent_config import AgentConfig

# # Import your existing modules (adjust paths as needed)
# from tools.llm_functions import CallAgent
# from prompts import get_prompt

# logger = logging.getLogger("livekit-agent")
# logger.setLevel(logging.INFO)

# async def entrypoint(ctx: JobContext):
#     """
#     Main entrypoint for the LiveKit agent
#     """
#     # Load configuration
#     config = AgentConfig()
    
#     phone_number = ctx.job.metadata if ctx.job.metadata else None
#     logger.info(f"🚀 Agent connecting to room {ctx.room.name} to dial {phone_number}")

#     await ctx.connect()

#     # Handle SIP participant setup if phone number provided
#     if phone_number is not None:
#         participant_name = f"phone_user-{phone_number}"
        
#         # Create SIP participant
#         await ctx.api.sip.create_sip_participant(
#             api.CreateSIPParticipantRequest(
#                 room_name=ctx.room.name,
#                 sip_trunk_id=config.outbound_trunk_id,
#                 sip_call_to=phone_number,
#                 participant_identity=participant_name,
#             )
#         )

#         # Wait for participant and check call status
#         participant = await ctx.wait_for_participant(identity=participant_name)

#         start_time = perf_counter()
#         while perf_counter() - start_time < 30:  # 30 second timeout
#             call_status = participant.attributes.get("sip.callStatus")
            
#             if call_status == "active":
#                 logger.info("📞 Call answered by user")
#                 break
#             elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
#                 logger.info("❌ User rejected the call")
#                 await ctx.shutdown()
#                 return
#             elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
#                 logger.info("❌ User unavailable")
#                 await ctx.shutdown()
#                 return
            
#             await asyncio.sleep(0.1)

#     # Initialize custom AI components
#     custom_llm = CustomLLM(**config.get_llm_config())
#     custom_asr = CustomASR(**config.get_asr_config())
#     custom_tts = CustomTTS(**config.get_tts_config())
    
#     # Create your call agent with custom LLM
#     agent = CallAgent(instructions=get_prompt(), ctx=ctx)
    
#     # Create agent session with custom components
#     session = AgentSession(
#         stt=custom_asr.get_stt(),
#         llm=custom_llm.get_llm(),
#         tts=custom_tts.get_tts(),
#         vad=silero.VAD.load(
#             min_silence_duration=config.vad_min_silence_duration,
#             min_speech_duration=config.vad_min_speech_duration,
#             max_buffered_speech=config.vad_max_buffered_speech,
#         ),
#     )

#     # Event handlers for conversation logging
#     def on_conversation_item_added(event):
#         async def handle_conversation_item():
#             item = event.item
            
#             if item.role == "user":
#                 logger.info(f"[USER] {item.text_content}")
                        
#             elif item.role == "assistant":
#                 logger.info(f"[AGENT] {item.text_content}")
        
#         asyncio.create_task(handle_conversation_item())
    
#     session.on("conversation_item_added")(on_conversation_item_added)

#     # Start the agent session
#     await session.start(
#         agent=agent,
#         room=ctx.room,
#     )

# def prewarm_fnc(proc: JobProcess):
#     """Prewarm function to load VAD model"""
#     proc.userdata["vad"] = silero.VAD.load()

# if __name__ == "__main__":
#     logger.info("🎯 Starting Modular LiveKit Agent")
#     logger.info("🔧 Components: OpenAI LLM + Deepgram ASR + ElevenLabs TTS")
    
#     # You can customize agent name here
#     agent_name = "modular-agent-1"
    
#     cli.run_app(
#         WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             agent_name=agent_name,
#             prewarm_fnc=prewarm_fnc,
#         )
#     )



import asyncio
import logging
from time import perf_counter

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    JobProcess
)
from livekit.plugins import silero
from livekit.plugins import deepgram  # Added deepgram import for TTS

# Import custom components
from custom_llm import CustomLLM
from custom_groq_llm import CustomGroqLLM  # New Groq LLM import
from custom_asr import CustomASR
from custom_tts import CustomTTS
from agent_config import AgentConfig

# Import your existing modules (adjust paths as needed)
from tools.llm_functions import CallAgent
# from prompts import get_prompt

def get_prompt(timezone: str = "Asia/Kolkata") -> str:
    return "You are a helpful telephony assistant"

logger = logging.getLogger("livekit-agent")
logger.setLevel(logging.INFO)

def create_llm_instance(config: AgentConfig):
    """
    Factory function to create the appropriate LLM instance
    based on configuration
    """
    llm_config = config.get_llm_config()
    
    if config.is_groq_enabled():
        logger.info(f"🦙 Using Groq LLM with model: {llm_config['model']}")
        return CustomGroqLLM(**llm_config)
    else:
        logger.info(f"🤖 Using OpenAI LLM with model: {llm_config['model']}")
        return CustomLLM(**llm_config)

async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent
    """
    # Load configuration
    config = AgentConfig()
    
    phone_number = ctx.job.metadata if ctx.job.metadata else None
    logger.info(f"🚀 Agent connecting to room {ctx.room.name} to dial {phone_number}")
    logger.info(f"🔧 LLM Provider: {config.llm_provider.upper()}")

    await ctx.connect()

    # Handle SIP participant setup if phone number provided
    if phone_number is not None:
        participant_name = f"phone_user-{phone_number}"
        
        # Create SIP participant
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=config.outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_name,
            )
        )

        # Wait for participant and check call status
        participant = await ctx.wait_for_participant(identity=participant_name)

        start_time = perf_counter()
        while perf_counter() - start_time < 30:  # 30 second timeout
            call_status = participant.attributes.get("sip.callStatus")
            
            if call_status == "active":
                logger.info("📞 Call answered by user")
                break
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_REJECTED:
                logger.info("❌ User rejected the call")
                await ctx.shutdown()
                return
            elif participant.disconnect_reason == rtc.DisconnectReason.USER_UNAVAILABLE:
                logger.info("❌ User unavailable")
                await ctx.shutdown()
                return
            
            await asyncio.sleep(0.1)

    # Initialize AI components based on configuration
    custom_llm = create_llm_instance(config)  # This will create either OpenAI or Groq LLM
    custom_asr = CustomASR(**config.get_asr_config())
    custom_tts = CustomTTS(**config.get_tts_config())
    
    # Create your call agent with the selected LLM
    agent = CallAgent(instructions=get_prompt(), ctx=ctx)
    
    # Create agent session with custom components
    session = AgentSession(
        stt=custom_asr.get_stt(),
        llm=custom_llm.get_llm(),  # This works for both OpenAI and Groq
        # tts=custom_tts.get_tts(),
        tts = deepgram.TTS(),
        vad=silero.VAD.load(
            min_silence_duration=config.vad_min_silence_duration,
            min_speech_duration=config.vad_min_speech_duration,
            max_buffered_speech=config.vad_max_buffered_speech,
        ),
    )

    # Event handlers for conversation logging
    def on_conversation_item_added(event):
        async def handle_conversation_item():
            item = event.item
            
            if item.role == "user":
                logger.info(f"[USER] {item.text_content}")
                        
            elif item.role == "assistant":
                logger.info(f"[AGENT] {item.text_content}")
        
        asyncio.create_task(handle_conversation_item())
    
    session.on("conversation_item_added")(on_conversation_item_added)

    # Start the agent session
    await session.start(
        agent=agent,
        room=ctx.room,
    )

def prewarm_fnc(proc: JobProcess):
    """Prewarm function to load VAD model"""
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    # Load config to show which LLM is being used
    try:
        config = AgentConfig()
        llm_info = f"{config.llm_provider.upper()}"
        if config.is_groq_enabled():
            llm_info += f" ({config.groq_model})"
        else:
            llm_info += f" ({config.llm_model})"
    except Exception as e:
        llm_info = "Configuration Error"
        logger.error(f"Configuration error: {e}")
    
    logger.info("🎯 Starting Modular LiveKit Agent")
    logger.info(f"🔧 Components: {llm_info} LLM + Deepgram ASR + ElevenLabs TTS")
    
    # You can customize agent name here
    agent_name = "modular-agent-1"
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
            prewarm_fnc=prewarm_fnc,
        )
    )