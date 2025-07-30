# src/llm/gemini_llm.py
import os
from google import generativeai as genai
from src.ai.base_llm import BaseLLM
from src.common.config import CONFIG

class GeminiLLM(BaseLLM):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.__llm = genai.GenerativeModel(CONFIG["gemini"]["model"]) 

    def invoke(self, messages: list[dict]):
        # Extract only user-visible content to pass as a prompt
        user_messages = [msg["content"] for msg in messages if msg["role"] in {"user", "system"}]
        prompt = "\n\n".join(user_messages)

        response = self.__llm.generate_content(
            prompt,
            generation_config={
                "temperature": CONFIG["llm"]["gemini"]["temperature"],
                "top_p": CONFIG["llm"]["gemini"]["top_p"],
                "max_output_tokens": CONFIG["llm"]["gemini"]["max_output_tokens"],
            }
        )

        return response.text
