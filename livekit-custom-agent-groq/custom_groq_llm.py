import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, Literal
from dotenv import load_dotenv
from groq import Groq

from livekit.agents import llm
from livekit.agents.llm import (
    LLM, 
    LLMStream,
    ChatContext,
    ChatChunk,
    ChatMessage,
    ChatRole,
    ChoiceDelta
)

load_dotenv()

logger = logging.getLogger("groq-llm")

class GroqLLMStream(LLMStream):
    """
    Custom LLMStream implementation for Groq
    Compatible with LiveKit v1.1.4
    """
    
    def __init__(
        self,
        llm: LLM,
        groq_client: Groq,
        generation_params: dict,
        *,
        chat_ctx: ChatContext,
        tools: Any = None,  # Use Any instead of specific type
        conn_options: Dict[str, Any],
    ):
        super().__init__(
            llm=llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._groq_client = groq_client
        self._generation_params = generation_params
    
    async def _run(self) -> None:
        """
        Main method that runs the Groq streaming and emits ChatChunk events
        """
        try:
            logger.info("🦙 Starting Groq streaming")
            
            # Create streaming completion
            completion = self._groq_client.chat.completions.create(**self._generation_params)
            
            chunk_id = f"groq_{id(completion)}"
            
            # Stream the response
            content_buffer = ""
            chunk_count = 0
            
            for chunk in completion:
                chunk_count += 1
                
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        content_buffer += content
                        
                        # Create LiveKit ChatChunk for v1.1.4
                        chat_chunk = self._create_chat_chunk(chunk_id, content)
                        
                        # Send the chunk through LiveKit's event system
                        await self._event_ch.asend(chat_chunk)
                        
                        # Allow other async tasks to run
                        await asyncio.sleep(0)
            
            logger.info(f"🦙 Groq streaming completed: {len(content_buffer)} chars, {chunk_count} chunks")
            
            if len(content_buffer) == 0:
                logger.warning("🦙 Empty response from Groq")
                # Send a fallback response
                fallback_chunk = self._create_chat_chunk(
                    chunk_id,
                    "I apologize, but I'm having trouble generating a response right now."
                )
                await self._event_ch.asend(fallback_chunk)
            
        except Exception as e:
            logger.error(f"🦙 Error in Groq streaming: {e}")
            logger.exception("🦙 Full traceback:")
            
            # Send error response
            error_chunk = self._create_chat_chunk(
                f"groq_error_{id(e)}",
                "I'm sorry, I'm experiencing technical difficulties. Please try again."
            )
            await self._event_ch.asend(error_chunk)
    
    def _create_chat_chunk(self, chunk_id: str, content: str) -> ChatChunk:
        """Create a ChatChunk compatible with LiveKit v1.1.4"""
        try:
            # Create choice delta - content might need to be a list in v1.1.4
            try:
                choice_delta = ChoiceDelta(content=content, role="assistant")
            except Exception:
                # Try with list content if string fails
                choice_delta = ChoiceDelta(content=[content], role="assistant")
            
            # Create choice object dynamically (since Choice class isn't available)
            choice = type('Choice', (), {
                'delta': choice_delta,
                'index': 0
            })()
            
            # Create ChatChunk with required 'id' field for v1.1.4
            return ChatChunk(
                id=chunk_id,  # Use 'id' instead of 'request_id' for v1.1.4
                choices=[choice]
            )
            
        except Exception as e:
            logger.error(f"🦙 Error creating ChatChunk: {e}")
            # Ultra-minimal fallback
            return type('ChatChunk', (), {
                'id': chunk_id,
                'choices': [type('Choice', (), {
                    'delta': type('Delta', (), {
                        'content': content,
                        'role': 'assistant'
                    })()
                })()]
            })()


class GroqLLM(LLM):
    """
    Groq LLM implementation for LiveKit v1.1.4
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        super().__init__()
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY environment variable is required")
            self.client = Groq()
        
        logger.info(f"🦙 Groq LLM initialized with model: {model}")
    
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: Dict[str, Any] = None,
        tools: Any = None,  # Use Any instead of specific type
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[str, Literal["auto", "required", "none"]] | None = None,
    ) -> LLMStream:
        """
        Required method by LiveKit LLM interface
        Returns an LLMStream that handles the streaming response
        """
        if conn_options is None:
            conn_options = {"max_retry": 3, "retry_interval": 2.0, "timeout": 10.0}
            
        logger.info("🦙 chat method called")
        
        # Convert LiveKit messages to Groq format
        # In v1.1.4, ChatContext uses 'items' instead of 'messages'
        lk_messages = getattr(chat_ctx, 'messages', None) or getattr(chat_ctx, 'items', [])
        messages = self._convert_messages(lk_messages)
        
        if not messages:
            logger.warning("🦙 No messages to process")
            # Create a minimal stream that will send an error
            return GroqLLMStream(
                llm=self,
                groq_client=self.client,
                generation_params={},
                chat_ctx=chat_ctx,
                tools=tools,
                conn_options=conn_options,
            )
        
        # Log the last user message for debugging
        user_messages = [m for m in messages if m.get('role') == 'user']
        if user_messages:
            logger.info(f"🦙 Last user message: {user_messages[-1].get('content', '')[:100]}...")
        
        # Generation parameters
        generation_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        
        logger.info(f"🦙 Creating LLMStream with model: {self.model}")
        
        return GroqLLMStream(
            llm=self,
            groq_client=self.client,
            generation_params=generation_params,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
    
    def _convert_messages(self, lk_messages) -> List[Dict[str, str]]:
        """Convert LiveKit messages to Groq format"""
        groq_messages = []
        
        logger.debug(f"🦙 Converting {len(lk_messages)} messages")
        
        for i, msg in enumerate(lk_messages):
            try:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # Handle LiveKit ChatMessage objects
                    role_str = str(msg.role).lower()
                    
                    if 'user' in role_str:
                        role = 'user'
                    elif 'assistant' in role_str:
                        role = 'assistant'
                    elif 'system' in role_str:
                        role = 'system'
                    else:
                        role = 'user'  # Default fallback
                    
                    # Handle content - it might be a string or list
                    content = ""
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        # Handle list of content items (for multimodal)
                        content_parts = []
                        for item in msg.content:
                            if isinstance(item, str):
                                content_parts.append(item)
                            elif hasattr(item, 'text'):
                                content_parts.append(str(item.text))
                            else:
                                content_parts.append(str(item))
                        content = " ".join(content_parts)
                    else:
                        content = str(msg.content)
                    
                    content = content.strip()
                    if content:  # Only add non-empty messages
                        groq_messages.append({
                            "role": role,
                            "content": content
                        })
                        logger.debug(f"🦙 Message {i}: {role} -> '{content[:50]}...'")
                        
                elif isinstance(msg, dict):
                    # Handle dict format
                    role = msg.get("role", "user")
                    content = str(msg.get("content", "")).strip()
                    if content:
                        groq_messages.append({
                            "role": role,
                            "content": content
                        })
                        
                else:
                    # Handle string or other formats
                    content = str(msg).strip()
                    if content:
                        groq_messages.append({
                            "role": "user",
                            "content": content
                        })
                        
            except Exception as e:
                logger.warning(f"🦙 Error processing message {i}: {e}")
                continue
        
        logger.info(f"🦙 Converted to {len(groq_messages)} valid messages")
        return groq_messages


class CustomGroqLLM:
    """
    Wrapper class that follows your existing CustomLLM pattern
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        reasoning_effort: str = "medium"  # Kept for compatibility but not used
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        
        # Verify API key is set
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        logger.info(f"🦙 Custom Groq LLM initialized with model: {model}")
    
    def get_llm(self):
        """
        Returns configured Groq LLM instance
        This matches your existing CustomLLM interface
        """
        return GroqLLM(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def update_config(self, **kwargs):
        """
        Update LLM configuration parameters
        This matches your existing CustomLLM interface
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self.get_llm()


# Test function
async def test_groq_llm():
    """Test the Groq LLM implementation"""
    try:
        groq_llm = GroqLLM()
        
        # Check what ChatRole values are actually available
        print(f"🔍 ChatRole type: {ChatRole}")
        print(f"🔍 Available ChatRole values: {getattr(ChatRole, '__args__', 'No __args__')}")
        
        # Check ChatContext constructor
        import inspect
        print(f"🔍 ChatContext signature: {inspect.signature(ChatContext.__init__)}")
        
        # Create a test chat context using 'items' parameter as expected by v1.1.4
        messages = [
            ChatMessage(role="user", content=["Hello, can you respond briefly?"])
        ]
        
        try:
            # Try with 'items' parameter (v1.1.4 format)
            chat_ctx = ChatContext(items=messages)
            print("✅ Created ChatContext with items parameter")
        except Exception as e1:
            try:
                # Try positional argument
                chat_ctx = ChatContext(messages)
                print("✅ Created ChatContext with positional argument")
            except Exception as e2:
                print(f"❌ Failed both methods: {e1}, {e2}")
                return False
        
        print("🦙 Testing Groq LLM...")
        
        # Use the chat method like LiveKit does
        llm_stream = groq_llm.chat(chat_ctx=chat_ctx)
        
        response = ""
        async for chunk in llm_stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        print(content, end="", flush=True)
                        response += content
        
        print(f"\n🦙 Test completed. Response length: {len(response)}")
        return True
        
    except Exception as e:
        print(f"🦙 Test failed: {e}")
        logger.exception("🦙 Test exception:")
        return False

if __name__ == "__main__":
    # Set API key for testing
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY not set. Please set it in your .env file:")
        print("GROQ_API_KEY=your_actual_groq_api_key_here")
        print("\nSkipping test...")
    else:
        asyncio.run(test_groq_llm())