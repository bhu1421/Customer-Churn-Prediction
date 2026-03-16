#!/usr/bin/env python3
"""
Customer Churn Prediction - Deployment Runner
This script helps you run both the FastAPI backend and Streamlit frontend.
"""

import os
import subprocess
import sys
import time


def run_command(command, name):
    """Run a command in the background."""
    print(f"Starting {name}...")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd(),
        )
        return process
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None


def main():
    print("Customer Churn Prediction - Deployment")
    print("=" * 50)

    required_files = ["model.pkl", "scaler.pkl", "features.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"[ERROR] Missing model files: {missing_files}")
        print("Please run: python train_and_save_model.py")
        sys.exit(1)

    print("[OK] Model files found")

    fastapi_cmd = "uvicorn app:app --reload --host 0.0.0.0 --port 8000"
    fastapi_process = run_command(fastapi_cmd, "FastAPI server")

    if fastapi_process:
        print("[OK] FastAPI server started on http://localhost:8000")
        print("   - API docs: http://localhost:8000/docs")
        print("   - Health check: http://localhost:8000/health")

    time.sleep(3)

    streamlit_cmd = "streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
    streamlit_process = run_command(streamlit_cmd, "Streamlit app")

    if streamlit_process:
        print("[OK] Streamlit app started on http://localhost:8501")

    print("\n[OK] Both servers are running.")
    print("Open your browser to: http://localhost:8501")
    print("API available at: http://localhost:8000")
    print("\nPress Ctrl+C to stop both servers")

    try:
        while True:
            if fastapi_process and fastapi_process.poll() is not None:
                print("[ERROR] FastAPI server stopped unexpectedly.")
                break
            if streamlit_process and streamlit_process.poll() is not None:
                print("[ERROR] Streamlit app stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        if fastapi_process and fastapi_process.poll() is None:
            fastapi_process.terminate()
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
        print("[OK] Servers stopped")


if __name__ == "__main__":
    main()
