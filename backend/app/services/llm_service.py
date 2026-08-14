import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class LlmService:
    _client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    _model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    @classmethod
    def generate(cls, prompt: str) -> str:
        response = cls._client.models.generate_content(
            model=cls._model,
            contents=prompt,
        )

        return response.text
