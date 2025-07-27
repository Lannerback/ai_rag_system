# src/llm/gemini_llm.py
import os
from google import generativeai as genai
from src.ai.base_llm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.__llm = genai.GenerativeModel(os.getenv("GEMINI_MODEL")) 

    def invoke(self, messages: list[dict]):
        # Extract only user-visible content to pass as a prompt
        user_messages = [msg["content"] for msg in messages if msg["role"] in {"user", "system"}]
        prompt = "\n\n".join(user_messages)

        #TODO: Handle system prompt and other configurations from a configuration file
        response = self.__llm.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 500,
            }
        )

        return response.text
