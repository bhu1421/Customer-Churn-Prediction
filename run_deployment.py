import os
import subprocess
import sys
import time

import requests

from config import FEATURES_PATH, MODEL_PATH


API_BASE_URL = "http://127.0.0.1:8000"


def run_command(command: list[str], name: str, env: dict[str, str] | None = None):
    print(f"Starting {name}...")
    try:
        return subprocess.Popen(
            command,
            env=env,
        )
    except Exception as exc:
        print(f"Error starting {name}: {exc}")
        return None


def wait_for_api(timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(1)

    return False


def stop_process(process: subprocess.Popen | None, name: str) -> None:
    if not process or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    print(f"[OK] {name} stopped")


def model_files_available() -> bool:
    return MODEL_PATH.exists() and FEATURES_PATH.exists()


if __name__ == "__main__":
    print("Customer Churn Prediction - Deployment")
    print("=" * 50)

    if not model_files_available():
        print("[ERROR] Missing model files in the model_files folder.")
        print("Please run: python train_and_save_model.py")
        sys.exit(1)

    print("[OK] Model files found")

    fastapi_process = run_command(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        "FastAPI server",
    )
    if not fastapi_process:
        sys.exit(1)

    if not wait_for_api():
        print("[ERROR] FastAPI server did not become ready in time.")
        stop_process(fastapi_process, "FastAPI server")
        sys.exit(1)

    print("[OK] FastAPI server started on http://localhost:8000")
    print("   - API docs: http://localhost:8000/docs")
    print("   - Health check: http://localhost:8000/health")

    streamlit_env = os.environ.copy()
    streamlit_env["API_BASE_URL"] = API_BASE_URL

    streamlit_process = run_command(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        "Streamlit app",
        env=streamlit_env,
    )
    if not streamlit_process:
        stop_process(fastapi_process, "FastAPI server")
        sys.exit(1)

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
        stop_process(streamlit_process, "Streamlit app")
        stop_process(fastapi_process, "FastAPI server")
        print("[OK] Servers stopped")
