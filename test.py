import os
from dotenv import load_dotenv

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Check if the API key is loaded correctly
if TOGETHER_API_KEY is None:
    print("Error: API key not found!")
else:
    print(f"API Key: {TOGETHER_API_KEY}")
