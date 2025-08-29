#!/usr/bin/env python3
"""
Complete diagnostic to check ALL LiveKit signatures for v1.1.4
"""

import inspect
from livekit.agents.llm import (
    LLM, 
    LLMStream,
    ChatContext,
    ChatChunk,
    ChatMessage,
    ChatRole,
    ChoiceDelta
)

def check_all_signatures():
    """Check all the signatures we need"""
    print("🔍 Complete LiveKit v1.1.4 Signature Analysis")
    print("=" * 60)
    
    # 1. ChatRole values
    print(f"1️⃣ ChatRole:")
    print(f"   Type: {ChatRole}")
    print(f"   Values: {getattr(ChatRole, '__args__', 'No __args__')}")
    print()
    
    # 2. ChatMessage constructor
    print(f"2️⃣ ChatMessage:")
    try:
        sig = inspect.signature(ChatMessage.__init__)
        print(f"   Constructor: {sig}")
        
        # Try to create a sample
        try:
            msg = ChatMessage(role="user", content=["test"])
            print(f"   ✅ Sample created: role=string, content=list works")
        except Exception as e:
            print(f"   ❌ Sample failed: {e}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    # 3. ChatContext constructor
    print(f"3️⃣ ChatContext:")
    try:
        sig = inspect.signature(ChatContext.__init__)
        print(f"   Constructor: {sig}")
        
        # Try to create a sample
        try:
            msg = ChatMessage(role="user", content=["test"])
            ctx = ChatContext(items=[msg])
            print(f"   ✅ Sample created with items=[...]")
            print(f"   Available attributes: {[attr for attr in dir(ctx) if not attr.startswith('_')]}")
        except Exception as e:
            print(f"   ❌ Sample failed: {e}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    # 4. LLMStream constructor
    print(f"4️⃣ LLMStream:")
    try:
        sig = inspect.signature(LLMStream.__init__)
        print(f"   Constructor: {sig}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    # 5. LLM.chat method
    print(f"5️⃣ LLM.chat method:")
    try:
        sig = inspect.signature(LLM.chat)
        print(f"   Method: {sig}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    # 6. ChatChunk constructor
    print(f"6️⃣ ChatChunk:")
    try:
        sig = inspect.signature(ChatChunk.__init__)
        print(f"   Constructor: {sig}")
        
        # Try to create a sample
        try:
            chunk = ChatChunk(id="test", choices=[])
            print(f"   ✅ Sample created with id and choices")
        except Exception as e:
            print(f"   ❌ Sample failed: {e}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    # 7. ChoiceDelta constructor
    print(f"7️⃣ ChoiceDelta:")
    try:
        sig = inspect.signature(ChoiceDelta.__init__)
        print(f"   Constructor: {sig}")
        
        # Try to create a sample
        try:
            delta = ChoiceDelta(content="test", role="assistant")
            print(f"   ✅ Sample created with content=string, role=string")
        except Exception as e1:
            try:
                delta = ChoiceDelta(content=["test"], role="assistant")
                print(f"   ✅ Sample created with content=list, role=string")
            except Exception as e2:
                print(f"   ❌ Both samples failed: {e1}, {e2}")
    except Exception as e:
        print(f"   ❌ Inspection failed: {e}")
    print()
    
    print("🎯 Summary for Implementation:")
    print("=" * 40)
    print("Based on the above, the correct implementation should use:")
    print("- ChatMessage(role='user', content=['text'])")
    print("- ChatContext(items=[messages])")
    print("- Check LLMStream.__init__ signature carefully")
    print("- Check LLM.chat signature carefully")
    print("- ChatChunk(id='...', choices=[...])")
    print("- ChoiceDelta with appropriate content format")

if __name__ == "__main__":
    check_all_signatures()