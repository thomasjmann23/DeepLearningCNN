#!/usr/bin/env python3
"""
Launch script for Fashion-MNIST Streamlit Application
Run this script to start the web application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Set streamlit configuration
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Launch streamlit app
    print("Starting Fashion-MNIST CNN Classifier Web Application...")
    print("Open your browser and go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.call([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")

if __name__ == "__main__":
    main()