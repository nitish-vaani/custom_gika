"""
Earkart Agent class definition.
Contains the main agent logic and function tools.
"""

import json
import os
from typing import Any, AsyncIterable
from livekit import rtc
from livekit.agents import (Agent, function_tool, RunContext, llm, ChatContext, ChatMessage)
from livekit.agents import ModelSettings, FunctionTool
from utils.hungup_idle_call import hangup
from utils.utils import load_prompt
from utils.preprocess_text_before_tts import preprocess_text
from utils.gpt_inferencer import LLMPromptRunner
from .config_manager import config_manager
from .call_handlers import CallState
from .database_helpers import insert_call_end_async
from .transcript_manager import transcript_manager
from .logging_config import get_logger
from .rag_connector import enrich_with_rag

logger = get_logger(__name__)

class EarkartAgent(Agent):
    """Main Earkart agent class with all business logic and function tools"""
    
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
        call_state: CallState,
        prompt_path: str,
    ):
        super().__init__(
            instructions=load_prompt(prompt_path, full_path=True)
        )
        self.name = name
        self.appointment_time = appointment_time
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        self.llm_obj = LLMPromptRunner(api_key=config_manager.get_openai_api_key())
        self.call_state = call_state
        self._seen_results = set()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """Custom LLM node implementation"""
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Custom TTS node with text preprocessing"""
        async def cleaned_text():
            async for chunk in text:
                yield preprocess_text(chunk)

        async for frame in Agent.default.tts_node(self, cleaned_text(), model_settings):
            yield frame

    def set_participant(self, participant: rtc.RemoteParticipant):
        """Set the participant for this agent session"""
        self.participant = participant

    async def record_call_end(self, end_reason: str):
        """Record call end in database asynchronously with optimized queuing"""
        if self.call_state.call_end_recorded or not self.call_state.call_started:
            return
            
        try:
            self.call_state.call_end_recorded = True
            operation_id = await insert_call_end_async(
                self.call_state.room_name,
                end_reason
            )
            logger.info(f"Queued call end recording: {operation_id} - {end_reason}")
        except Exception as e:
            logger.error(f"Failed to queue call end recording: {e}")

    async def on_enter(self):
        """Called when agent enters the conversation"""
        await self.session.say(
            text="नमस्ते, मैं सुमित बोल रहा हूँ EarKart से. बताइए मैं आपकी कैसे help कर सकता हूँ?", allow_interruptions=True
        )
        agent_name = self.__class__.__name__
        
        # Import here to avoid circular imports
        from .data_entities import UserData
        userdata: UserData = self.session.userdata
        
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes(
                {"agent": agent_name}
            )

    @function_tool
    async def search_earkart_knowledge_base(self, context: RunContext, query: str):
        """
        Lookup EarKart knowledge base if extra information is needed for user's query. This method searches documents related to Earkart hering aid servicing, pricing, locations etc.
        """
        all_results = await enrich_with_rag(query)
        # Filter out previously seen results
        new_results = [
            r for r in all_results if r not in self._seen_results
        ]
        
        # If we don't have enough new results, clear the seen results and start fresh
        if len(new_results) == 0:
            return f"No new context found for query: {query}."
        else:
            new_results = new_results[:2]  # Take top 2 new results

        self._seen_results.update(new_results)

        context = ""
        for i, res in enumerate(new_results):
            context = context + "\n context " + str(i) + ":" + res + "\n"

        return new_results

    @function_tool
    async def validate_customer_details(self, ctx: RunContext):
        """Validate customer details by extracting entities from conversation"""
        entities = [
            ('Name', 'What is the name Of the User'),
            ('Mobile_Number', "What is the users mobile number?")
        ]
        from utils.entity_extractor_dynamic_prompt import generate_prompt_to_get_entities_from_transcript
        prompt = generate_prompt_to_get_entities_from_transcript(
            transcript=transcript_manager.get_transcript(), 
            fields=entities
        )
        content = self.llm_obj.run_prompt(prompt)
        
        # Clean up JSON response
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            content = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction response: {e}")
            return "I apologize, but I had trouble processing your information. Could you please repeat the key details?"
        
        # Check for missing information
        not_mentioned_keys = [key for key, val in content.items() if val.get('value') == 'Not Mentioned']
        if not_mentioned_keys:
            ask_about = "\n".join(f"{key}: {value}" for key, value in entities if key in not_mentioned_keys)
            return f"""Casually ask user about following missing informations: "{ask_about}". You can say 'Sorry, I missed asking X, please provide these details.'"""

        return "Noted"                        

    @function_tool()
    async def end_call(self, ctx: RunContext, current_language: str):
        """
        This method hungup/cut/end the ongoing call. Call this method when user request to cut the call or you want to exit the call.
        
        current_language: strictly either "Hindi" Or "English"
        """
        participant_id = self.participant.identity if self.participant else 'unknown'
        logger.info(f"Agent initiated call end for {participant_id}")

        await self.record_call_end("Call ended")

        # Wait for current speech to finish
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        if  "English" == current_language:
            end_call_msg = "Thank you so much for calling Earkart. Wish you good day ahead"
        else:
            end_call_msg = "EarKart को call करने के लिए बहुत-बहुत धन्यवाद"

        await self.session.say(text=end_call_msg)
        await hangup()
        return "Noted"
    
    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Handle answering machine detection"""
        participant_id = self.participant.identity if self.participant else 'unknown'
        logger.info(f"Detected answering machine for {participant_id}")
        
        await self.record_call_end("Answering machine detected")
        await hangup()
        return "Noted"
    
    # @function_tool()
    # async def get_service_pricing(self, ctx: RunContext):
    #     """Get pricing information for services"""
    #     # This can be expanded with actual pricing logic
    #     logger.info("Service pricing requested")
    #     return "Let me get you the latest pricing information for our services."

def create_agent(name: str, appointment_time: str, dial_info: dict[str, Any], 
                        call_state: CallState, prompt_path: str) -> EarkartAgent:
    """Factory function to create a Earkart instance"""
    return EarkartAgent(
        name=name,
        appointment_time=appointment_time,
        dial_info=dial_info,
        call_state=call_state,
        prompt_path=prompt_path
    )
