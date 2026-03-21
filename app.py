"""
Root-level app.py — Hugging Face Spaces entry point.
Hugging Face looks for app.py in the root directory.
This file simply adds src/ to the path and launches the actual app.
"""
import sys
import os

# Add src/ to path so all imports in the Streamlit app work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Also set working directory to src/ so relative paths
# (like data/faiss_index) resolve correctly
os.chdir(os.path.join(os.path.dirname(__file__), "src"))

# Import and run the actual Streamlit app
exec(open(os.path.join(os.path.dirname(__file__), "src", "app.py")).read())
