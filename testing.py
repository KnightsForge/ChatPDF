import os

if 'GEMINI_API_KEY' in os.environ:
    print("GEMINI_API_KEY exists")
else:
    print("GEMINI_API_KEY does not exist")