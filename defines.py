import os
import os.path

MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-5-haiku",
    "anthropic/claude-3-haiku",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
]

PROXIES = os.environ.get("PROXIES", None)
API_KEYS : list[str] = os.environ.get("API_KEYS", "").split(",")
AUTHORIZATION_TOKEN : str = os.environ.get("AUTHORIZATION_TOKEN", "")
