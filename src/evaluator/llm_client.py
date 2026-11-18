"""Unified LLM Client supporting Google Gemini, OpenAI, and local mock fallback."""
import os
import json
import re
from typing import Optional, Dict, Any, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified LLM client interface with automatic JSON extraction and multi-provider support."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = api_key or os.getenv("OPENAI_API_KEY")
        self._gemini = None
        self._openai = None

        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self._gemini = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
            except Exception:
                self._gemini = None

        if self.openai_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=self.openai_key)
            except Exception:
                self._openai = None

    @property
    def is_live_llm_available(self) -> bool:
        """Returns True if a live LLM (Gemini or OpenAI) is ready."""
        return (self._gemini is not None and bool(self.gemini_key)) or (self._openai is not None and bool(self.openai_key))

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        """Generates raw plain text response."""
        # 1. Try Gemini
        if self._gemini and (self.provider == "gemini" or not self._openai):
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                resp = self._gemini.generate_content(full_prompt)
                if resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        # 2. Try OpenAI
        if self._openai and (self.provider == "openai" or not self._gemini):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                resp = self._openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        return ""

    def generate_structured(self, prompt: str, system_prompt: str, schema: Type[T]) -> Optional[T]:
        """Generates and parses structured Pydantic object from LLM response."""
        json_instruction = (
            f"\n\nYou MUST respond with valid JSON strictly conforming to the following JSON schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Do not include any introductory or concluding markdown outside the JSON block."
        )

        full_prompt = prompt + json_instruction
        raw_response = self.generate_text(prompt=full_prompt, system_prompt=system_prompt)

        if not raw_response:
            return None

        # Extract JSON from markdown code fences if present
        json_str = self._extract_json_string(raw_response)
        try:
            parsed_dict = json.loads(json_str)
            return schema.model_validate(parsed_dict)
        except Exception:
            return None

    @classmethod
    def _extract_json_string(cls, text: str) -> str:
        """Extracts JSON substring from markdown backticks or braces."""
        text = text.strip()
        # Look for ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        # Look for { ... } or [ ... ]
        brace_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if brace_match:
            return brace_match.group(1).strip()

        return text
