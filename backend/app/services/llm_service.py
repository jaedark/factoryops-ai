import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


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
    def generate_content(
        cls,
        contents,
        config: types.GenerateContentConfig | None = None,
    ):
        return cls._client.models.generate_content(
            model=cls._model,
            contents=contents,
            config=config,
        )

    @classmethod
    def generate(cls, prompt: str) -> str:
        response = cls.generate_content(
            contents=prompt,
        )

        return response.text
