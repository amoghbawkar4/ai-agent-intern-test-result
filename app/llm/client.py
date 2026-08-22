import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMClient:
    """Small wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        model: str | None = None,
    ):
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.model = (
            model
            or os.getenv(
                "OPENAI_MODEL",
                "gpt-5-mini",
            )
        )

        self.client = OpenAI(
            api_key=api_key
        )

    def generate(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        """Generate a text response."""

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
        )

        return response.output_text