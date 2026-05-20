"""
AegisLife AETHER — gemini_client.py
=====================================
Switched from Gemini to Groq (LLaMA 3.3 70B)
"""

import time
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ── Load .env (2 levels up from branch4_llm/llm/) ─────────────
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Groq client ───────────────────────────────────────────────
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"


def call_gemini(prompt: str, retries: int = 4) -> str:
    """
    Drop-in replacement for Gemini using Groq LLaMA 3.3 70B.
    Function name kept as call_gemini so nothing else breaks.
    """
    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=3000,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)
            retry_seconds = 15 * (attempt + 1)
            match = re.search(r"retry.*?(\d+)s|please wait (\d+)", error_str, re.IGNORECASE)
            if match:
                retry_seconds = int(match.group(1) or match.group(2)) + 3

            if "429" in error_str and attempt < retries - 1:
                print(f"[Groq] Rate limited. Waiting {retry_seconds}s before retry {attempt+2}/{retries}...")
                time.sleep(retry_seconds)
                continue

            return f"⚠️ AI Engine Error: {error_str[:300]}"

    return "⚠️ Maximum retries reached. Please wait 1 minute and try again."