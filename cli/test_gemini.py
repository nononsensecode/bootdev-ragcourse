import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client()
response = client.models.generate_content(
    model="gemma-4-31b-it",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)
print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response tokens: {response.usage_metadata.total_token_count}")
