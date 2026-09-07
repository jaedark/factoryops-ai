from google import genai
from google.genai import types

from backend.app.core.config import get_settings


class LlmService:
    _client = None
    _client_api_key: str | None = None

    @classmethod
    def _get_client(cls):
        api_key = get_settings().require_gemini_api_key()
        if cls._client is None or cls._client_api_key != api_key:
            cls._client = genai.Client(api_key=api_key)
            cls._client_api_key = api_key
        return cls._client

    @classmethod
    def reset_client(cls) -> None:
        cls._client = None
        cls._client_api_key = None

    @classmethod
    def generate_content(
        cls,
        contents,
        config: types.GenerateContentConfig | None = None,
    ):
        settings = get_settings()
        return cls._get_client().models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=config,
        )

    @classmethod
    def generate(cls, prompt: str) -> str:
        response = cls.generate_content(contents=prompt)
        return response.text
