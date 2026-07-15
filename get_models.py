import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    models = response.json()
    with open("models.json", "w") as f:
        json.dump(models, f, indent=4)
    print("Successfully saved models to models.json")
except Exception as e:
    print(f"Error fetching models: {e}")
