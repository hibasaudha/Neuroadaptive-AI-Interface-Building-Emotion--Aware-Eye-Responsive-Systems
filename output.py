import requests
import json
import time
from input import get_fused_emotion, get_realtime_face_emotion # Import the key function

# --- 1. CONFIGURATION & API SETUP ---
# NOTE: Make sure your multimodal_inputs.py is in the same directory.
API_KEY = "AIzaSyA2BfE8y37_3GIrDz45ByDZGzgQXYcHHLs" 
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"


# --- 4. GEMINI CHAT LOGIC ---

def get_cheer_up_response(user_query: str, fused_emotion: str) -> str:
    """
    Calls the Gemini API, using the FUSED emotional state as a System Instruction.
    """
    system_prompt = (
        f"You are a highly empathetic and supportive friend, coach, and companion. "
        f"Your primary goal is to respond to the user's message with warmth, validation, "
        f"and encouragement to cheer them up, based on their emotional state. "
        f"The user's FUSED emotional state is: **{fused_emotion}**. "
        f"Your response MUST be conversational, warm, and directly address the feeling expressed by the user and the FUSED emotional state. "
        f"Keep the response concise, under 80 words."
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {'Content-Type': 'application/json'}
    max_retries = 3
    
    for i in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if text:
                return text.strip()
            else:
                return "The model returned an empty response."

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and i < max_retries - 1:
                delay = 2**i + (time.monotonic() % 1)
                print(f"Rate limit hit (429), retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                return f"Gemini API Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred during API call: {e}"

    return "API call failed after multiple retries."


# --- 5. MAIN CHAT INTERFACE ---

def run_chat_app():
    """
    Runs the terminal-based multimodal chat application.
    """
    print("="*60)
    print("      Multimodal Empathetic Chatbot (Python Terminal App)      ")
    print("      Logic and Fusion imported from 'multimodal_inputs.py'      ")
    print("="*60)
    print("Companion: Hello! Tell me how you're feeling. (Type 'quit' or 'exit' to end)")
    print("-" * 60)

    while True:
        try:
            # We fetch the current face reading independently for display
            realtime_face = get_realtime_face_emotion()
            print(f"[CV PIPELINE] Current Real-Time Face Emotion: {realtime_face.upper()}")
            
            user_input = input("You: ")
            
            if user_input.lower() in ('quit', 'exit'):
                print("\nCompanion: Take care and have a wonderful day!")
                break
            
            if not user_input.strip():
                continue

            # STEP 1: FUSION LAYER (CALLING THE EXTERNAL MODULE)
            print("\n[ANALYSIS START] Running Multimodal Fusion Logic...")
            # This is where the fusion logic from the other file is executed
            fused_emotion = get_fused_emotion(user_input) 
            print(f"[ANALYSIS END] FUSED EMOTION STATE: === {fused_emotion} ===\n")
            
            # STEP 2: GEMINI RESPONSE
            print("Companion: (Thinking...)")
            cheer_up_response = get_cheer_up_response(user_input, fused_emotion)
            
            # STEP 3: OUTPUT
            print(f"Companion: {cheer_up_response}")
            print("-" * 60)

        except EOFError:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An application error occurred: {e}")
            break

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY is missing. Please edit the 'multimodal_client.py' file")
    else:
        run_chat_app()
