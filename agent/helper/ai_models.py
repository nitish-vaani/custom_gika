"""
AI model configuration and initialization.
Handles LLM, TTS, and other AI model setup.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from livekit.plugins import elevenlabs, deepgram, openai, cartesia, aws, silero
from .config_manager import config_manager
from .logging_config import get_logger

logger = get_logger(__name__)

def get_openai_llm():
    """Get properly configured OpenAI LLM"""
    api_key = config_manager.get_openai_api_key()
    
    try:
        # For project-specific keys, we might need to handle differently
        if api_key.startswith("sk-proj-"):
            logger.info("Using project-specific OpenAI API key")
        
        # Use basic configuration without unsupported parameters
        llm_instance = openai.LLM(
            # model="gpt-3.5-turbo",  # More reliable and supported model
            model="gpt-4o",  # Use gpt-4o for better performance
            api_key=api_key
        )
        
        logger.info("Successfully created OpenAI LLM instance")
        return llm_instance
        
    except Exception as e:
        logger.error(f"Failed to create OpenAI LLM: {e}")
        raise

def get_tts(config: Dict[str, Any]):
    """Get configured TTS instance based on config"""
    which_tts = config["TTS"]

    if which_tts == "cartesia":
        harry = "3dcaa773-fb1a-47f7-82a4-1bf756c4e1fb"
        carson = "4df027cb-2920-4a1f-8c34-f21529d5c3fe"
        happy_carson = "96c64eb5-a945-448f-9710-980abe7a514c"
        orion = "701a96e1-7fdd-4a6c-a81e-a4a450403599"
        polite_man = "ee7ea9f8-c0c1-498c-9279-764d6b56d189"
        american_voiceover_man = "7fe6faca-172f-4fd9-a193-25642b8fdb07"
        david = "da69d796-4603-4419-8a95-293bfc5679eb"
        ayush="791d5162-d5eb-40f0-8189-f19db44611d8"
        return cartesia.TTS(
            model="sonic-2-2025-03-07",
            voice=ayush,
            speed=0,
            language="hi",
            emotion=["positivity:highest", "curiosity:highest"],
        )
    
    if which_tts == "aws":
        return aws.TTS()

    if which_tts == "elevenlabs":
        @dataclass
        class VoiceSettings:
            stability: float
            similarity_boost: float
            style: float | None = None
            speed: float | None = 1.0
            use_speaker_boost: bool | None = False

        voice_setting = VoiceSettings(
            stability=0.5,
            speed=1.0,
            similarity_boost=0.6,
            style=0.0,
            use_speaker_boost=True,
        )
        eric_voice_id = "cjVigY5qzO86Huf0OWal"
        chinmay_voice_id = "xnx6sPTtvU635ocDt2j7"
        return elevenlabs.TTS(
            model="eleven_flash_v2_5", 
            voice_settings=voice_setting, 
            voice_id=chinmay_voice_id
        )

def get_stt_instance():
    """Get configured STT instance"""
    return deepgram.STT(
        model="nova-3", 
        language="multi"
    )

def get_vad_instance():
    """Get configured VAD instance"""
    return silero.VAD.load()