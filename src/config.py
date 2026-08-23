"""Shared configuration helpers for secrets and runtime settings."""

import os


def get_groq_api_key() -> str:
    """Return the Groq API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st

        secrets = getattr(st, "secrets", {})
        if isinstance(secrets, dict):
            api_key = secrets.get("GROQ_API_KEY")
            if api_key:
                return api_key
    except Exception:
        pass

    raise ValueError("GROQ_API_KEY not found. Set it as an environment variable or Streamlit secret.")