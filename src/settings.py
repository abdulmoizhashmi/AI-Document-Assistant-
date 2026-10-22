from __future__ import annotations

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def get_setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables or Streamlit Community Cloud secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        secret = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default

    return str(secret).strip() or default
