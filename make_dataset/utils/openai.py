import json
import os
import time
from typing import Any, Dict, List, Type

from openai import OpenAI
from pydantic import BaseModel

class OpenAIRequest:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def structured_request(
        system_prompt: str,
        user_prompt: str,
        text_format: Type[BaseModel],
        llm_model: str,
    ) -> BaseModel:
        response = OpenAIRequest.openai_client.responses.parse(
            model=llm_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=text_format,
        )
        return response.output_parsed