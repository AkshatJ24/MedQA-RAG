"""
Root-level app.py — Hugging Face Spaces entry point.
Hugging Face looks for app.py in the root directory.
This file simply adds src/ to the path and launches the actual app.
"""
import sys
import os

# Resolve the absolute base dir BEFORE any chdir() call,
# so paths stay correct regardless of where Streamlit runs from.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add src/ to path so all imports in the Streamlit app work correctly
sys.path.insert(0, os.path.join(_BASE_DIR, "src"))

# Also set working directory to src/ so relative paths
# (like data/faiss_index) resolve correctly
os.chdir(os.path.join(_BASE_DIR, "src"))

# Import and run the actual Streamlit app
exec(open(os.path.join(_BASE_DIR, "src", "app.py")).read())
