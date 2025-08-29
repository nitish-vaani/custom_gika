# #!/usr/bin/env python3
# """
# Simple test to verify Groq API connectivity
# Run this to check if your API key and setup are working
# """

# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# def test_groq_api():
#     """Test basic Groq API functionality"""
    
#     # Check API key
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         print("❌ GROQ_API_KEY environment variable not found!")
#         print("Please set it in your .env file:")
#         print("GROQ_API_KEY=your_actual_api_key_here")
#         return False
    
#     print(f"✅ Found API key: {api_key[:10]}...")
    
#     try:
#         # Initialize client
#         client = Groq(api_key=api_key)
#         print("✅ Groq client initialized")
        
#         # Test simple completion
#         print("🔄 Testing API call...")
        
#         completion = client.chat.completions.create(
#             model="openai/gpt-oss-20b",
#             messages=[
#                 {"role": "user", "content": "Say 'Hello from Groq!' and nothing else."}
#             ],
#             max_completion_tokens=50,
#             temperature=0.1
#         )
        
#         response = completion.choices[0].message.content
#         print(f"✅ API Response: {response}")
        
#         # Test streaming
#         print("🔄 Testing streaming...")
        
#         stream = client.chat.completions.create(
#             model="openai/gpt-oss-20b",
#             messages=[
#                 {"role": "user", "content": "Count from 1 to 3."}
#             ],
#             max_completion_tokens=50,
#             temperature=0.1,
#             stream=True
#         )
        
#         print("✅ Streaming response: ", end="")
#         for chunk in stream:
#             if chunk.choices[0].delta.content:
#                 print(chunk.choices[0].delta.content, end="", flush=True)
#         print()
        
#         print("🎉 All tests passed! Groq API is working correctly.")
#         return True
        
#     except Exception as e:
#         print(f"❌ Groq API test failed: {e}")
#         return False

# if __name__ == "__main__":
#     print("🦙 Testing Groq API connectivity...")
#     success = test_groq_api()
    
#     if success:
#         print("\n✅ Your Groq setup is working! You can now use it with LiveKit.")
#     else:
#         print("\n❌ Please fix the issues above before using Groq with LiveKit.")
#         print("\nCommon issues:")
#         print("1. Invalid API key - get one from https://console.groq.com/")
#         print("2. API key not in .env file")
#         print("3. Network connectivity issues")
#         print("4. Rate limiting (try again in a few seconds)")



#!/usr/bin/env python3
"""
Test different Groq models to find working ones
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def test_groq_models():
    """Test different Groq models to see which ones work"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found!")
        return False
    
    client = Groq(api_key=api_key)
    
    # List of models to test
    models_to_test = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant", 
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"  # Your original choice
    ]
    
    working_models = []
    
    for model in models_to_test:
        print(f"\n🔄 Testing model: {model}")
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Say 'Hello from " + model + "!' and nothing else."}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            response = completion.choices[0].message.content
            if response and response.strip():
                print(f"✅ {model}: {response.strip()}")
                working_models.append(model)
            else:
                print(f"⚠️  {model}: Empty response")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")
    
    print(f"\n🎯 Working models: {len(working_models)}")
    for model in working_models:
        print(f"  ✅ {model}")
    
    if working_models:
        print(f"\n🔧 Recommended model for your .env:")
        print(f"GROQ_MODEL={working_models[0]}")
        
        # Test streaming with the first working model
        print(f"\n🔄 Testing streaming with {working_models[0]}...")
        try:
            stream = client.chat.completions.create(
                model=working_models[0],
                messages=[
                    {"role": "user", "content": "Count slowly from 1 to 3, one number per line."}
                ],
                max_tokens=50,
                temperature=0.1,
                stream=True
            )
            
            print("✅ Streaming response:")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
    
    return len(working_models) > 0

if __name__ == "__main__":
    print("🦙 Testing Groq model availability...")
    success = test_groq_models()
    
    if not success:
        print("\n❌ No working models found. Please check:")
        print("1. Your API key permissions")
        print("2. Account status at https://console.groq.com/")
        print("3. Rate limits")